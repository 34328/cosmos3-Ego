# AR IT2V 纯视频预训练

设计与数据口径见 `docs/ar_it2v_v0.1/design.md`，实际训练回执见同目录 `experiment.md`。

- 训练入口：`bash cosmos3_ar_it2v/launch.sh`，复用官方 Cosmos3 launcher；正式必须 online W&B。
- 配方：`configs/ego100h.toml`，官方 Nano 起点、原始30fps连续视频、首帧+文本、无action。
- 模型：`model.py`、`attention.py`；数据：`dataset.py`；原生训练回调：`monitor.py`。
- 推理：`torchrun --standalone --nproc-per-node=1 -m cosmos3_ar_it2v.inference --help`；每次独立输出目录，禁止覆盖。
- 环境：`PYTHONPATH=<repo>:<repo>/packages/cosmos3`，`/home/lzh/miniconda3/envs/cosmos3/bin/python`。

原始100小时清单在 `/mnt/lzh/cosmos-EgoWAM/training_manifests/`；不复制共享数据。该分支不改动旧joint实验配方。
