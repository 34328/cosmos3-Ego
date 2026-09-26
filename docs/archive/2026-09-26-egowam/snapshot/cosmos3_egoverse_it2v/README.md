# Cosmos3 EgoVerse IT2V

Independent pure-video stage for EgoVerse. It reuses the `overfit_v0.0` dataset, prompt, and
temporal contract, but emits no action tensors and does not modify
`cosmos3_joint_video_hand_pose` or Cosmos core code.

Training initializes from `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464`, keeps the action
branch in the DCP but freezes it, and trains only `moe_gen`, `time_embedder`, `vae2llm`, and `llm2vae`.

Run `scripts/launch.sh`. Checkpoints are written to
`/mnt/lzh/cosmos/outputs/egoverse_it2v/train/v1` at iterations 300 and 600.

Run `scripts/run_replays.sh 300 600` after training. It evaluates the same four fixed long
segments used by the joint baseline and writes H.264 generated-vs-GT videos to
`/mnt/lzh/cosmos/outputs/egoverse_it2v/inference/v1/iter_*/replays/`.

The LR follow-up is `configs/train_v2.toml` / `scripts/launch_v2.sh`. It keeps the same data and
model contract, uses peak LR `8e-5` with the Nano vision-SFT optimizer schedule, and writes to
`outputs/egoverse_it2v/{train,inference}/v2`. Generate its replays with
`EGOVERSE_IT2V_VERSION=v2 scripts/run_replays.sh 300 600`.
