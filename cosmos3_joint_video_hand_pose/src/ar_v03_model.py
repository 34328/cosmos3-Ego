"""Single-pass joint diffusion forcing; V0.2 data and geometry stay unchanged."""

from __future__ import annotations

from itertools import accumulate

import attrs
import torch

from cosmos_framework.model.generator.omni_mot_causal_model import OmniMoTCausalModelConfig
from cosmos_framework.model.generator.utils.kv_cache import DualKVCache, TeacherForcingMemoryState

from .ar_model import OmniMoTCausalModel
from .ar_v02_layout import CONDITION_VIDEO, LAYOUT_VERSION
from .ar_v02_model import EgoVerseARV02Model
from .loss import whole_video_flow_loss


def wrist_channel_weights() -> list[float]:
    weights = [1.0] * 57
    weights[9:18] = [3.0] * 9
    weights[33:42] = [3.0] * 9
    return weights


@attrs.define(slots=False)
class EgoVerseARV03ModelConfig(OmniMoTCausalModelConfig):
    action_channel_weights: list[float] = attrs.field(factory=wrist_channel_weights)
    sigma_small: float = 0.02
    prefix_low_noise_enabled: bool = True
    sigma_hist_max: float = 0.1


class JointDiffusionForcingMemoryState(TeacherForcingMemoryState):
    """Reuse official three-way metadata, with no replay or retained layer K/V.

    All history is in the current differentiable GEN stream. The official
    memory adapter supplies text offsets and empty-cache placeholders only.
    Neither a second pass nor a cache write is needed for training.
    """

    def __init__(self, *, layouts, text_lengths, device, **kwargs):
        super().__init__(**kwargs)
        from .ar_v03_attention import JointDiffusionForcingAttention

        self.text_lengths = tuple(text_lengths)
        self.attention = JointDiffusionForcingAttention(layouts, device, text_lengths=text_lengths)

    def init(self, hidden_states, device):
        super().init(hidden_states, device)
        self.und_kv_offsets = torch.tensor(
            [0, *accumulate(self.text_lengths)], device=device, dtype=torch.int32
        )

    def read_for_layer(self, layer_idx):
        value = self._read_teacher_forcing_base_value(layer_idx)
        value.uses_rolling_gen_cache = False
        value.gen_attention_override = self.attention
        return value

    def write_for_layer(self, layer_idx, kv_to_store):
        # The attention output already holds the differentiable current K/V.
        # Do not retain or detach a second history copy between layers/steps.
        return None


class EgoVerseARV03Model(EgoVerseARV02Model):
    def __init__(self, config, **kwargs):
        weights = torch.as_tensor(config.action_channel_weights, dtype=torch.float32)
        if weights.shape != (57,) or not torch.isfinite(weights).all() or not (weights > 0).all():
            raise ValueError("action_channel_weights must be 57 finite positive values")
        if not 0 <= float(config.sigma_small) <= 1:
            raise ValueError("sigma_small must be in [0,1]")
        if not isinstance(config.prefix_low_noise_enabled, bool):
            raise ValueError("prefix_low_noise_enabled must be a bool")
        if not 0 < float(config.sigma_hist_max) <= 1:
            raise ValueError("sigma_hist_max must be in (0,1]")
        if kwargs.get("history_video_noise_prob", 0) != 0:
            raise ValueError("V0.3 uses diffusion forcing, not a second history-noise pass")
        super().__init__(config, **kwargs)

    def _validate_ar_config(self):
        cfg = self.config
        required = {
            "video_temporal_causal=True": bool(cfg.video_temporal_causal),
            "causal_training_strategy='diffusion_forcing'": cfg.causal_training_strategy == "diffusion_forcing",
            "joint_attn_implementation='three_way'": cfg.joint_attn_implementation == "three_way",
            "C4/K8": int(cfg.teacher_forcing_frames_per_chunk) == 4 and int(cfg.action_tokens_per_latent) == 8,
            "supervise_temporal_causal_actions=True": bool(cfg.supervise_temporal_causal_actions),
            "enable_moba=False": not bool(cfg.enable_moba),
            "CP1": int(cfg.parallelism.context_parallel_shard_degree) == 1,
            "no target-only replay": not bool(cfg.teacher_forcing_target_only_no_text_pass2),
        }
        missing = [name for name, ok in required.items() if not ok]
        if missing:
            raise ValueError("EgoVerseARV03Model requires " + ", ".join(missing))

    def pre_noise_memory_hook(self, packed_sequence, gen_data_clean, memory_info):
        self._validate_teacher_forcing_pack(packed_sequence)
        if "_tf_memory_state" in memory_info:
            raise ValueError("V0.3 training cannot replay a clean pass")
        return memory_info

    def build_memory_state(self, packed_seq, memory_info):
        if getattr(packed_seq, "joint_layout_version", None) != LAYOUT_VERSION:
            return OmniMoTCausalModel.build_memory_state(self, packed_seq, memory_info)
        if memory_info.get("dual_kv_cache") is not None:
            return OmniMoTCausalModel.build_memory_state(self, packed_seq, memory_info)
        self._validate_teacher_forcing_pack(packed_seq)
        net = self.net
        return JointDiffusionForcingMemoryState(
            layouts=packed_seq.joint_layouts,
            text_lengths=packed_seq.joint_text_lengths,
            device=packed_seq.vision.tokens[0].device,
            vision_token_shapes=packed_seq.vision.token_shapes,
            num_action_tokens_per_supertoken=packed_seq.num_action_tokens_per_supertoken,
            null_action_supertokens=packed_seq.null_action_supertokens,
            segment_idx=0,
            dual_kv_cache=[DualKVCache(gen_cache_size=2) for _ in range(net.num_hidden_layers)],
            num_kv_heads=net.num_kv_heads,
            head_dim=net.head_dim,
            detach_clean_kv=False,
            clamp_empty_varlen_kv=self.config.clamp_empty_varlen_kv,
            frames_per_chunk=4,
        )

    def denoise(self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        # Inference supplies JointKVCache; training supplies the single-pass
        # adapter above. Neither uses V0.2's compact noisy/replay path.
        return OmniMoTCausalModel.denoise(self, net, data_batch_packed, memory, video_temporal_causal)

    def _get_train_noise_level_vision(
        self, batch_size, is_image_batch, num_vision_latent_frames, resolutions=None, num_tokens=None, iteration=None
    ):
        if is_image_batch:
            raise ValueError("V0.3 requires paired video/action clips")
        layouts = self._joint_layouts
        n = max(len(x.boundaries) for x in layouts) + 1

        def repeat(values):
            if values is None or isinstance(values, str):
                return values
            return [value for value in values for _ in range(n)]

        # Native DF samples B*T_max values. Request one latent frame for each
        # independent chunk draw, then route it to all four frames in the
        # actual chunk. This preserves V0.2's distribution and RNG ordering.
        ts, sg = OmniMoTCausalModel._get_train_noise_level_vision(
            self, batch_size=batch_size * n, is_image_batch=False,
            num_vision_latent_frames=[1] * (batch_size * n),
            resolutions=repeat(resolutions), num_tokens=repeat(num_tokens), iteration=iteration,
        )
        ts, sg = ts.reshape(batch_size, n), sg.reshape(batch_size, n)
        step = self._ar_step
        step.prefix_low_noise = []
        step.prefix_low_noise_plan = None
        if self.config.prefix_low_noise_enabled:
            from .ar_v03_sigma import sample_prefix_low_noise, apply_prefix_low_noise, continuous_rf_timesteps

            if getattr(self, "_ar_gradient_accumulation", 1) != 1:
                raise ValueError("stateless prefix sampling requires grad_accum_iter=1")
            # CP1/CFGP1 are required by V0.3, so world rank is the DP rank.
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            plan = sample_prefix_low_noise(
                [len(x.boundaries) for x in layouts], seed=self.ar_seed,
                iteration=iteration, rank=rank, sigma_hist_max=self.config.sigma_hist_max,
            )
            sg = apply_prefix_low_noise(sg, plan)
            ts = torch.where(plan.mask.to(ts.device), continuous_rf_timesteps(sg, self.rectified_flow_video), ts)
            step.prefix_low_noise_plan = plan
            step.prefix_low_noise = plan.metadata
        step.video_chunk_sigmas = sg
        width = max(num_vision_latent_frames)
        timesteps, sigmas = ts.new_zeros(batch_size, width), sg.new_zeros(batch_size, width)
        for i, layout in enumerate(layouts):
            roles, chunks, _ = layout.video_metadata(device=sg.device)
            timesteps[i, :len(roles)] = torch.where(roles == CONDITION_VIDEO, 0, ts[i, chunks])
            sigmas[i, :len(roles)] = torch.where(roles == CONDITION_VIDEO, 0, sg[i, chunks])
        self._ar_step.chunk_ids = torch.arange(n, device=sg.device)
        return timesteps, sigmas

    def _get_train_noise_level_action(self, batch_size, iteration=None):
        # Always consume the original official action draws first. Prefix
        # sampling uses its private generator, so suffix values and the later
        # epsilon draws retain the original global RNG sequence.
        timesteps, sigmas = super()._get_train_noise_level_action(batch_size, iteration=iteration)
        step = self._ar_step
        plan = getattr(step, "prefix_low_noise_plan", None)
        if plan is None:
            return timesteps, sigmas
        from .ar_v03_sigma import apply_prefix_low_noise, continuous_rf_timesteps

        step.action_sigmas = apply_prefix_low_noise(step.action_sigmas, plan)
        current = step.action_sigmas[:, 1:2]
        timesteps = torch.where(plan.mask[:, 1:2].to(timesteps.device),
                               continuous_rf_timesteps(current, self.rectified_flow_action), timesteps)
        return timesteps, current

    def _add_noise_to_input(self, *args, **kwargs):
        result = super()._add_noise_to_input(*args, **kwargs)
        # Ephemeral references for the short-test callback's read-only audit.
        # The enclosing official training_step clears _ar_step on exit.
        self._ar_step.noised_video_sigmas = result.sigmas_vision
        self._ar_step.noised_action_sigmas = result.sigmas_action
        return result

    def _compute_whole_losses(self, out_net, packed, noised, timesteps, is_image_batch):
        """Keep V0.2's global sample reduction and raw field logs, add weights.

        V0.2 has no per-channel loss extension hook. This versioned override
        changes the action reducer only; official Trainer/backward stay intact.
        """
        from .ar_v03_loss import whole_action_flow_loss

        cfg, rf = self.config, self.config.rectified_flow_training_config
        if is_image_batch or not cfg.vision_gen or not cfg.action_gen or cfg.sound_gen or getattr(cfg, "lidar_gen", False):
            raise ValueError("V0.3 whole loss requires paired video/action only")
        if rf.loss_scale != 1.0 or rf.action_loss_weight != 1.0:
            raise ValueError("V0.3 objective is L_video + L_action")
        for kind in ("und", "gen"):
            if out_net.get(f"lbl_metadata_{kind}") is not None and getattr(getattr(cfg, "lbl", None), f"coeff_{kind}", 0):
                raise ValueError("V0.3 does not include auxiliary load balancing")
        if packed.vision is None or packed.action is None:
            raise ValueError("V0.3 pack must retain both modalities")
        n = len(out_net["preds_action"])
        if len(out_net["preds_vision"]) != n or len(packed.sample_lens) != n:
            raise ValueError("one video/action item per logical sample required")
        if len(packed.action.raw_action_dim) != n or any(d is None or int(d) != 57 for d in packed.action.raw_action_dim):
            raise ValueError("explicit raw_action_dim=57 required")
        if self._current_hand_visibility is None:
            raise RuntimeError("action loss reached without synchronized hand visibility")

        def video_weight(index, frames, reference):
            ts = timesteps[index, :frames] if timesteps.ndim > 1 else timesteps[index]
            return self.rectified_flow_video.train_time_weight(ts, self.tensor_kwargs_fp32)

        _, video = whole_video_flow_loss(
            pred=out_net["preds_vision"], target=noised.vt_target_vision,
            condition_mask=packed.vision.condition_mask, time_weight=video_weight,
        )
        _, action = whole_action_flow_loss(
            pred=out_net["preds_action"], target=noised.vt_target_action,
            condition_mask=packed.action.condition_mask, visibility=self._current_hand_visibility,
            valid_mask=packed.action.action_valid_mask, mask_out_of_fov=False,
            collect_field_metrics=True, channel_weights=cfg.action_channel_weights,
        )
        window = getattr(self, "_ar_loss_window", None)
        if window is None or window.complete:
            if getattr(self, "_ar_gradient_accumulation", 1) != 1:
                raise RuntimeError("preplan all microbatch counts before accumulation")
            counts = torch.stack((video["active_samples"].sum(), action["active_samples"].sum()))[None]
            self.begin_ar_loss_window(counts, device=counts.device)
            window = self._ar_loss_window
        backward, stats = window.reduce(
            video["per_sample_losses"], video["active_samples"], action["per_sample_losses"], action["active_samples"],
            action_weight=rf.action_loss_weight,
        )
        self._ar_backward_loss = backward
        self._last_visibility_loss_metrics = {
            name + "_loss": values.sum() * window.world_size * window.microsteps / window.global_counts[1].clamp_min(1)
            for name, values in action.get("field_per_sample_losses", {}).items()
        }
        v, a = stats["video_contribution"] * window.world_size, stats["action_contribution"] * window.world_size
        logged_loss = backward / window.microsteps
        return logged_loss, {
            "train_objective_numerator": logged_loss.detach(),
            "train_objective_denominator": torch.ones_like(logged_loss.detach()),
            "flow_matching_loss_vision": v, "flow_matching_loss_action": a,
            "flow_matching_loss_vision_per_instance": video["unweighted_per_sample_losses"].detach(),
            "egoverse_global_video_samples": stats["global_video_samples"],
            "egoverse_global_action_samples": stats["global_action_samples"],
        }
