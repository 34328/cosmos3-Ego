"""Versioned wrist-local hand codec for fixed-camera rigid action deltas."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn

REPRESENTATION = 'fixed_camera_wrist_local_delta_latent_v1'
INPUT_FRAME = 'current_frame_wrist_local'
ARCHITECTURE = '60-64-SiLU-32-SiLU-15 / 15-32-SiLU-64-SiLU-60'
PCA_ARCHITECTURE = 'centered-physical-PCA60-to-15-v1'


def tensor_hash(tensors):
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        t = tensor.detach().cpu().to(torch.float32).contiguous()
        h.update(name.encode()); h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


def stats_hash(payload):
    return tensor_hash(dict(input_mean=payload['state_dict']['mean'], input_std=payload['state_dict']['std'], latent_mean=payload['latent_mean'], latent_std=payload['latent_std']))


def validate_source_provenance(fit, split):
    """Require source-window identities as well as declared split/hash labels."""
    if fit.get('split') != split:
        raise ValueError(f'codec source must be {split}-only')
    if fit.get('input_frame') != INPUT_FRAME or fit.get('representation') != REPRESENTATION:
        raise ValueError('codec source must use the wrist-local representation')
    ids, windows = fit.get('episode_ids', []), fit.get('source_windows', [])
    if not ids or len(ids) != len(set(ids)) or not windows:
        raise ValueError('codec requires unique episode IDs and source windows')
    if {w.get('episode') for w in windows} != set(ids):
        raise ValueError('codec source windows do not match episode provenance')
    for w in windows:
        if (not w.get('window') or type(w.get('start')) is not int or w['start'] < 0
                or type(w.get('span')) is not int or w['span'] < 2):
            raise ValueError('invalid codec source window')
    for key in ('manifest_sha256', 'episodes_sha256'):
        value = fit.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing codec source {key}')
    sources = fit.get('source_hashes', {})
    for key in ('train_episodes', 'train_segments', 'heldout_episodes', 'heldout_segments'):
        value = sources.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing audited codec source hash: {key}')
    if fit['episodes_sha256'] != sources[split + '_episodes']:
        raise ValueError('codec episode source hash differs from audit')


def validate_source_pair(train, heldout):
    """Compare the same audit while allowing distinct train/heldout CSV files."""
    validate_source_provenance(train, 'train')
    validate_source_provenance(heldout, 'heldout')
    if (train['manifest_sha256'] != heldout['manifest_sha256']
            or train['source_hashes'] != heldout['source_hashes']):
        raise ValueError('codec validation provenance differs from fit')
    if set(train['episode_ids']) & set(heldout['episode_ids']):
        raise ValueError('codec heldout/train episode overlap')


class FrozenFixedCameraHandCodec(nn.Module):
    """encode(q)->absolute standardized z; caller differences z only after encoding.

    Validation tools alone may opt into allow_unvalidated. Runtime requires a
    passed sidecar tied to the exact checkpoint SHA256, never legacy fallback.
    """
    representation = REPRESENTATION
    coordinate_system = 'fixed-camera'
    input_frame = INPUT_FRAME

    def __init__(self, checkpoint, *, expected_sha256=None, validation_report=None, allow_unvalidated=False):
        super().__init__()
        path = Path(checkpoint)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('fixed-camera codec checkpoint hash mismatch')
        p = torch.load(path, map_location='cpu', weights_only=True)
        expected = dict(schema_version=2, representation=REPRESENTATION, input_frame=INPUT_FRAME,
                        coordinate_system='fixed-camera', chunk_size=4, action_tokens_per_latent=8, units='m')
        for key, value in expected.items():
            if p.get(key) != value:
                raise ValueError(f'fixed-camera codec incompatible {key}: {p.get(key)!r}')
        self.architecture = p.get('architecture')
        if self.architecture not in (ARCHITECTURE, PCA_ARCHITECTURE):
            raise ValueError('unsupported codec architecture')
        fit = p.get('fit', {})
        validate_source_provenance(fit, 'train')
        if fit.get('split') != 'train' or not fit.get('episode_ids') or fit.get('sample_count', 0) < 2:
            raise ValueError('codec requires fitted train-only statistics/provenance')
        if p.get('side') not in ('right', 'left'):
            raise ValueError('codec must identify right or left hand')
        for key in ('data_sha256', 'manifest_sha256', 'episodes_sha256'):
            if not isinstance(fit.get(key), str) or len(fit[key]) != 64:
                raise ValueError(f'missing fitted train {key}')
        if p.get('stats_sha256') != stats_hash(p):
            raise ValueError('codec fitted statistics hash mismatch')
        state = p['state_dict']
        expected_keys = {'mean','std','wrist_local_data_binding'}
        if self.architecture == PCA_ARCHITECTURE:
            expected_keys |= {'components'}
        else:
            self.encoder = nn.Sequential(nn.Linear(60,64), nn.SiLU(), nn.Linear(64,32), nn.SiLU(), nn.Linear(32,15))
            self.decoder = nn.Sequential(nn.Linear(15,32), nn.SiLU(), nn.Linear(32,64), nn.SiLU(), nn.Linear(64,60))
            expected_keys |= {'encoder.'+k for k in self.encoder.state_dict()} | {'decoder.'+k for k in self.decoder.state_dict()}
        if set(state) != expected_keys:
            raise ValueError('codec state_dict architecture keys mismatch')
        # Stored alongside learned tensors, absent from camera-axis checkpoints.
        # Changing top-level metadata alone cannot migrate old learned weights.
        binding = torch.tensor(list(bytes.fromhex(fit['data_sha256'])), dtype=torch.uint8)
        if state['wrist_local_data_binding'].dtype != torch.uint8 or not torch.equal(state['wrist_local_data_binding'], binding):
            raise ValueError('codec wrist-local training data binding mismatch')
        if self.architecture == PCA_ARCHITECTURE:
            components = state['components'].float()
            if (components.shape != (15,60) or not torch.isfinite(components).all()
                    or not torch.allclose(components @ components.T, torch.eye(15), atol=1e-5, rtol=1e-5)
                    or not torch.equal(state['std'], torch.ones(60))):
                raise ValueError('PCA requires orthonormal components and unscaled physical input')
            self.register_buffer('components', components)
        else:
            self.encoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('encoder.')}, strict=True)
            self.decoder.load_state_dict({k[8:]:v for k,v in state.items() if k.startswith('decoder.')}, strict=True)
        for key, tensor, size in [('input_mean',state['mean'],60), ('input_std',state['std'],60), ('latent_mean',p['latent_mean'],15), ('latent_std',p['latent_std'],15)]:
            if tensor.shape != (size,) or not torch.isfinite(tensor).all() or ('std' in key and not (tensor > 0).all()):
                raise ValueError(f'invalid codec {key}')
            self.register_buffer(key,tensor.float())
        if any(not torch.isfinite(t).all() for t in self.parameters()):
            raise ValueError('non-finite codec weights')
        if not allow_unvalidated:
            report_path = Path(validation_report) if validation_report else path.with_suffix('.validation.json')
            if not report_path.is_file():
                raise ValueError('fixed-camera codec lacks passed heldout validation sidecar')
            report = json.loads(report_path.read_text())
            if (report.get('checkpoint_sha256') != digest or report.get('passed') is not True
                    or report.get('representation') != REPRESENTATION or report.get('input_frame') != INPUT_FRAME
                    or report.get('schema_version') != 2):
                raise ValueError('fixed-camera codec validation failed or stale')
            if set(report.get('heldout_episode_ids', [])) & set(fit['episode_ids']) or not report.get('heldout_episode_ids'):
                raise ValueError('codec validation split leakage/empty heldout')
        self.checkpoint_sha256 = digest
        if not allow_unvalidated:
            provenance = report.get('heldout_provenance', {})
            validate_source_provenance(provenance, 'heldout')
            if set(provenance.get('episode_ids', [])) != set(report['heldout_episode_ids']):
                raise ValueError('codec validation episode provenance mismatch')
            validate_source_pair(fit, provenance)
            try:
                metrics, gates = report['metrics'], report['thresholds']
                pairs = [(metrics['reconstruction']['mean_mm'], gates['reconstruction_mean_mm']),
                         (metrics['reconstruction']['p95_mm'], gates['reconstruction_p95_mm'])]
                valid = all(math.isfinite(v) and math.isfinite(g) and 0 <= v <= g and g > 0 for v, g in pairs)
                invariant = metrics['reanchor_invariance']
                valid = valid and gates['reconstruction_mean_mm'] <= 5 and gates['reconstruction_p95_mm'] <= 15
                valid = valid and report['heldout_sample_count'] > 0 and metrics['trajectory_count'] > 0
                valid = valid and invariant['passed'] is True and invariant['latent_exact'] is True and invariant['no_reencode'] is True
                valid = valid and invariant['full_chunks'] >= 100 and invariant['tail_frames'] in (8, 16, 24)
                valid = valid and 0 <= invariant['max_point_error_m'] <= 1e-5
                valid = valid and invariant['nonzero_wrist_rotation'] is True and invariant['nonzero_camera_rotation'] is True
                if self.architecture == PCA_ARCHITECTURE:
                    repeated = metrics['repeat_decode']
                    valid = valid and report.get('architecture') == PCA_ARCHITECTURE
                    valid = valid and repeated['passed'] is True and repeated['repeats'] >= 10
                    valid = valid and repeated['max_abs_difference_m'] == 0
            except (KeyError, TypeError):
                valid = False
            if not valid:
                raise ValueError('codec validation metrics do not pass declared gates')
        self.checkpoint_path = str(path.resolve())
        self.metadata = {k:v for k,v in p.items() if k not in ('state_dict','latent_mean','latent_std')}
        self.requires_grad_(False); self.eval()

    @torch.no_grad()
    def encode(self, points):
        if points.shape[-2:] != (20,3) or not torch.isfinite(points).all():
            raise ValueError('expected finite wrist-local offsets [...,20,3]')
        if points.device != self.input_mean.device:
            raise ValueError('move codec to input device with codec.to(device) before encode')
        if points.dtype not in (torch.float32, torch.float64):
            raise ValueError('physical codec inputs require float32 or float64')
        flat = points.reshape(*points.shape[:-2],60).to(self.input_mean.dtype)
        raw = ((flat-self.input_mean) @ self.components.T if self.architecture == PCA_ARCHITECTURE
               else self.encoder((flat-self.input_mean)/self.input_std))
        return ((raw-self.latent_mean)/self.latent_std).to(points.dtype)

    @torch.no_grad()
    def decode(self, z):
        if z.shape[-1:] != (15,) or not torch.isfinite(z).all():
            raise ValueError('expected finite absolute codec latent [...,15]')
        if z.device != self.input_mean.device:
            raise ValueError('move codec to input device with codec.to(device) before decode')
        if z.dtype not in (torch.float32, torch.float64):
            raise ValueError('physical codec inputs require float32 or float64')
        raw = z.to(self.input_mean.dtype)*self.latent_std+self.latent_mean
        flat = (raw @ self.components + self.input_mean if self.architecture == PCA_ARCHITECTURE
                else self.decoder(raw)*self.input_std+self.input_mean)
        return flat.reshape(*z.shape[:-1],20,3).to(z.dtype)


class FrozenFixedCameraHandAE15(FrozenFixedCameraHandCodec):
    """Compatibility entry for explicitly MLP-only callers."""

    def __init__(self, checkpoint, **kwargs):
        super().__init__(checkpoint, **kwargs)
        if self.architecture != ARCHITECTURE:
            raise ValueError('MLP AE entry cannot load a PCA architecture')
