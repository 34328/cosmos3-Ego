# AR IT2V 纯视频预训练

完整segment与动态packing见 `docs/ar_it2v_v0.2/design.md`。旧最长97帧随机短窗口训练已停止，历史记录保留在 `docs/ar_it2v_v0.1/`。新正式训练等待用户确认。

- 训练入口：`bash cosmos3_ar_it2v/launch.sh`，复用官方 Cosmos3 launcher；正式必须 online W&B。
- 新入口：`configs/ego100h_full_segments.toml`，首帧+文本、无action，官方token预算75008。正常段保留全部原帧；仅超预算单段按批准的90%→50%阶梯均匀保留帧，记录真实帧率和时间跨度。旧`ego100h.toml`仅供复现。
- 已选配方：4个空闲节点HSDP8×4，1500步/save500；lr1e-4→3e-5、warmup100/cycle1500、weight_decay0.01。正式启动仍等GPU短测结果后用户确认。
- 模型：`model.py`、`attention.py`；数据：`dataset.py`；原生训练回调：`monitor.py`。
- 推理：`torchrun --standalone --nproc-per-node=1 -m cosmos3_ar_it2v.inference --help`；每次独立输出目录，禁止覆盖。
- 环境：`PYTHONPATH=<repo>:<repo>/packages/cosmos3`，`/home/lzh/miniconda3/envs/cosmos3/bin/python`。

原始100小时清单在 `/mnt/lzh/cosmos-EgoWAM/training_manifests/`；不复制共享数据。该分支不改动旧joint实验配方。
