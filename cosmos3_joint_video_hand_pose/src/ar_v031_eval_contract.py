"""Strict V0.3.1 provenance; generic DCP readiness remains official."""
from pathlib import Path
import hashlib

MODEL_TARGET = "cosmos3_joint_video_hand_pose.src.ar_v031_model.EgoVerseARV031Model"
MODEL_VERSION = "ar_v0.3.1"
MILESTONES = (500, 1000, 2000, 3000)
PREFIX_LOSS_DENOMINATOR = "v030_full_valid"
PREFIX_LOSS_MASK_SCOPE = "numerator_only"
GROUP_MSE_CALLBACK_TARGET = "cosmos3_joint_video_hand_pose.src.ar_v031_config.ARV031GroupMSECallback"
REFERENCE_RELATIVE = "outputs/joint_video_hand_pose/ar_v0_3/formal_prefix_uniform_t273_20261001T182754Z/config.yaml"
REFERENCE_SHA256 = "e7c5ab0150300247f139405470c16cbd668a8f22a8ce0cecf97795bda952ae66"

def repository(path=__file__):
    for parent in Path(path).resolve().parents:
        if (parent / "packages/cosmos3").is_dir() and (parent / "cosmos3_joint_video_hand_pose").is_dir():
            return parent
    raise ValueError("actual Cosmos repository cannot be located")

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _flatten(value, path=()):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(_flatten(child, path + (key,)))
        return result
    return {path: value}

def validate_training_config(config, reference):
    """Allow exactly the ablation target and experiment identity/budget changes."""
    expected = _flatten(reference)
    actual = _flatten(config)
    allowed = {
        ("model", "_target_"), ("model", "config", "mask_prefix_loss"),
        ("job", "group"), ("job", "name"), ("trainer", "max_iter"),
        ("trainer", "callbacks", "egoverse_loss_wandb", "_target_"),
    }
    changed = {path for path in expected.keys() | actual.keys()
               if path not in expected or path not in actual or expected[path] != actual[path]}
    extra = changed - allowed
    if extra:
        raise ValueError("V0.3.1 recipe differs from V0.3.0: " + repr(sorted(extra)))
    cfg = config["model"]["config"]
    if (config["model"]["_target_"] != MODEL_TARGET
            or cfg.get("mask_prefix_loss") is not True
            or cfg.get("prefix_low_noise_enabled") is not True
            or cfg.get("sigma_hist_max") != 0.1
            or cfg.get("causal_training_strategy") != "diffusion_forcing"
            or config["trainer"]["max_iter"] != 3000
            or config["trainer"]["callbacks"]["egoverse_loss_wandb"]["_target_"] != GROUP_MSE_CALLBACK_TARGET
            or config["job"]["group"] != "ar_v0_3_1"
            or not isinstance(config["job"]["name"], str) or not config["job"]["name"]):
        raise ValueError("not the approved V0.3.1 numerator-only prefix mask with original denominator and 3000-step recipe")
    return {"model_target": MODEL_TARGET, "model_version": MODEL_VERSION,
            "mask_prefix_loss": True, "prefix_loss_denominator": PREFIX_LOSS_DENOMINATOR,
            "prefix_loss_mask_scope": PREFIX_LOSS_MASK_SCOPE,
            "group_mse_callback_target": GROUP_MSE_CALLBACK_TARGET,
            "recipe_changed_paths": [".".join(x) for x in sorted(changed)]}

def validate_checkpoint_snapshot(snapshot, checkpoint):
    """Read only config/DCP metadata; require this actual run's complete save."""
    import re
    import yaml
    from cosmos3_joint_video_hand_pose.scripts.ar_v03_eval_followup import checkpoint_ready

    snapshot, checkpoint = Path(snapshot).resolve(), Path(checkpoint).resolve()
    match = re.fullmatch(r"iter_(\d{9})", checkpoint.parent.name)
    if (snapshot.name != "config.yaml" or not snapshot.is_file()
            or checkpoint.name != "model" or not match
            or checkpoint.parent.parent != snapshot.parent / "checkpoints"
            or int(match.group(1)) not in MILESTONES):
        raise ValueError("V0.3.1 requires this run's config.yaml and step500/1000/2000/3000 official model DCP")
    reference = repository() / REFERENCE_RELATIVE
    if not reference.is_file() or sha256(reference) != REFERENCE_SHA256:
        raise ValueError("immutable V0.3.0 recipe reference changed")
    config = yaml.safe_load(snapshot.read_text())
    info = validate_training_config(config, yaml.safe_load(reference.read_text()))
    ready = checkpoint_ready(snapshot.parent, int(match.group(1)))
    if ready.get("ready") is not True:
        raise ValueError("official V0.3.1 save is incomplete: " + repr(ready))
    info.update(training_snapshot=str(snapshot), training_snapshot_sha256=sha256(snapshot),
                checkpoint_metadata_sha256=sha256(checkpoint / ".metadata"),
                recipe_reference_snapshot=str(reference), recipe_reference_snapshot_sha256=REFERENCE_SHA256,
                checkpoint_readiness=ready)
    return config, info

def validate_inference_config(config, snapshot_config):
    """Compare the actual TOML config before loading weights; no model instantiation."""
    import copy
    import yaml
    from tempfile import TemporaryDirectory
    from cosmos_framework.utils.lazy_config import LazyConfig

    # Trainer overwrites config.yaml with this exact serializer, which also
    # expands LazyCall defaults. Use the same representation, not type metadata
    # from the distinct serialization.to_yaml format.
    with TemporaryDirectory(prefix="ar_v031_infer_") as temp:
        path = Path(temp) / "config.yaml"
        LazyConfig.save_yaml(config, str(path))
        actual = yaml.safe_load(path.read_text())
    wanted = copy.deepcopy(snapshot_config)
    overrides = {
        ("parallelism", "data_parallel_shard_degree"): 1,
        ("parallelism", "data_parallel_replicate_degree"): 1,
        ("parallelism", "context_parallel_shard_degree"): 1,
        ("parallelism", "enable_inference_mode"): True,
        ("activation_checkpointing", "mode"): "none",
    }
    for (section, key), value in overrides.items():
        if actual["model"]["config"][section][key] != value:
            raise ValueError("missing documented inference-only model override")
        wanted["model"]["config"][section][key] = value
    for subtree in ("model", "dataloader_train", "optimizer", "scheduler"):
        left, right = _flatten(actual[subtree]), _flatten(wanted[subtree])
        changed = {path for path in left.keys() | right.keys()
                   if path not in left or path not in right or left[path] != right[path]}
        if changed:
            raise ValueError("inference TOML disagrees with trained snapshot: " +
                             subtree + " " + repr(sorted(changed)))
    return {"inference_config_matches_training_snapshot": True,
            "inference_only_overrides": [".".join(path) for path in overrides]}
