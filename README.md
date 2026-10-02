# Cosmos3 AR IT2V

将 CMD Stage 1 的因果视频训练方法迁移到 **Cosmos3-Nano**，使用 EgoVerse 约 100 小时视频做领域 AR 继续预训练。输入为**首帧图像＋文本**，输出为逐块生成的视频；本项目没有 action 模态。

## 当前配方

- 原始 30fps 连续视频，stride 1，不抽帧、不倍速。
- 复用原始 train/test 与文本动作段，窗口不跨标注边界；最长 97 帧，支持短段。
- 完整 clip 连续 VAE 编码；latent 分块 `[1,4,4,…]`，local16 总窗口，无永久首帧 sink。
- 单遍 Diffusion Forcing：首帧干净且不计 loss，未来块分别采样 σ、块内共享，所有未来块计算 flow MSE。
- 官方 Nano 起点，GEN-only 参数训练，保留官方 Trainer、optimizer、scheduler、DCP 和 W&B。
- 本轮为多步 AR 预训练，不包含 Context-matched distillation。

## 入口

| 内容 | 路径 |
|---|---|
| 设计与验收口径 | [docs/ar_it2v_v0.1/design.md](docs/ar_it2v_v0.1/design.md) |
| 训练状态与实验回执 | [docs/ar_it2v_v0.1/experiment.md](docs/ar_it2v_v0.1/experiment.md) |
| 配置 | [cosmos3_ar_it2v/configs/ego100h.toml](cosmos3_ar_it2v/configs/ego100h.toml) |
| 模型与块因果注意力 | [model.py](cosmos3_ar_it2v/model.py)、[attention.py](cosmos3_ar_it2v/attention.py) |
| 数据 | [dataset.py](cosmos3_ar_it2v/dataset.py) |
| 官方训练启动入口 | [launch.sh](cosmos3_ar_it2v/launch.sh) |
| 逐块推理 | [inference.py](cosmos3_ar_it2v/inference.py) |
| 测试 | `tests/test_ar_it2v_*.py` |
| 官方 Cosmos3 框架 | [packages/cosmos3/AGENTS.md](packages/cosmos3/AGENTS.md) |

远端工作目录 `/mnt/lzh/cosmos-ar-it2v`，分支 `ar-it2v-pretrain`。原始清单引用 `/mnt/lzh/cosmos-EgoWAM/training_manifests/` 下的 100 小时数据，视频与权重直接使用共享存储，不复制进源码目录。

环境使用 `/home/lzh/miniconda3/envs/cosmos3/bin/python`，`PYTHONPATH` 同时包含仓库根和 `packages/cosmos3`。正式启动前遵循 [AGENTS.md](AGENTS.md) 检查资源、已有任务与配置；正式训练 online W&B，真实进度以实验记录和 API 核实结果为准。
