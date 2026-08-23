# EgoVerse Cosmos 3 Action Artifact Contract v1

This directory is the single machine-readable binding point for the EgoVerse
57D action representation, frozen hand codec, and action normalizers.

## Current status

`manifest.json` is finalized with `training_ready=true`.

- The canonical seed-1 left/right MLP-AE-15 codec artifacts are ready and are
  referenced by path and SHA256 instead of being copied.
- The final state and future-delta normalizers were computed from all 32,355
  train samples under the finalized TODO-4 temporal contract and are bound by
  path, method, parameters, and SHA256.
- Stable parameter files live in `normalizers/`; the generator and full
  per-time audit remain in `tmp/cosmos3_todo6_normalizer_20260815/`.
- The older 121-frame normalization report is recorded as analysis-only. It
  must not be loaded for training because that window design was withdrawn.

## Why this exists

The same checkpoint must always use the same:

- 57D field order and temporal semantics;
- left/right codec weights and their embedded input/latent statistics;
- state normalizer;
- future-delta normalizer.

A shape-compatible but different artifact can silently change the physical
meaning of the model inputs and outputs. Training, resume, and inference must
therefore verify hashes and fail on any mismatch.

## Validation

Run the training-readiness integrity check:

```bash
python cosmos3_joint_video_hand_pose/artifacts/cosmos3_action_contract/v1/validate_manifest.py --require-training-ready
```

## TODO-6 result

1. State statistics use train split slot-0 tokens only.
2. Future-delta statistics use train split future tokens only.
3. Both use reversible `piecewise_asinh_rot`, `beta=1`, q01/q99 affine bounds,
   and no forward clamp.
4. The full audit and generator are in
   `tmp/cosmos3_todo6_normalizer_20260815/`.
5. Store the finalized manifest SHA256 and component hashes in every training
   checkpoint and reject mismatches on resume or inference.
