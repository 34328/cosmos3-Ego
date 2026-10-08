# Decoded-oracle geometry loss on v0.6

This isolated recipe applies the frozen MLP-AE-15 decoded-point objective to
the latest v0.6 joint video/action code at `d90219e`. It retains the B3
frame-delta rigid poses, v0.6 temporal attention mask, 57D action, CP1/FSDP8,
and 75K pack cap. The hand latent slices `[18:33]` and `[42:57]` are unchanged
by B3, so the decoder and point target use the existing codec contract.

The dataset adds wrist-local `[T,20,3]` point targets only for this recipe.
The training model recovers clean action from the native flow tensors, decodes
both hand latents with frozen weights, and adds a sigma-gated, warmup-ramped
decoded-oracle point loss. Bone and velocity weights are zero. The extra
fields pass through the current action transform and packing collator.

Run with `scripts/launch_geometry_b_decode_v0_6.sh` after confirming the
remote Python, eight GPUs, input Zarr paths, checkpoint/cache paths, and free
output space. Set `PYTHON_BIN` and `TORCHRUN_BIN` if the remote environment
differs from the historical `/home/lzh/miniconda3/envs/cosmos3` location.
This creates a separate `geometry/geometry_b_decode_v0_6` output directory.

Local verification is limited to Python compilation, TOML parsing and diff
checks because this macOS environment has no PyTorch. CUDA forward/backward,
distributed behavior and training outcomes still require a target-side smoke
check. The geometry optimization was previously paused after codec studies;
this recipe provides compatibility for evaluation, not evidence of a gain.
