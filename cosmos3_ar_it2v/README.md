# AR IT2V 纯视频预训练

首帧图像＋segment文本输入，输出只有视频。当前 V0.2 使用完整连续动作段和官方动态 Packer，已完成1500步；实际状态、checkpoint与结果见 [实验记录](../docs/ar_it2v_v0.2/experiment.md)。

| 入口 | 用途 |
|---|---|
| `configs/ego100h_full_segments.toml` / `config.py` | 完整segment，75008 token，HSDP8×4，1500步/save500，峰值lr1e-4→3e-5，wd0.01 |
| `dataset.py` / `dataloader.py` | Zarr段边界与文本、真实长度、超预算策略、官方 packing 适配 |
| `model.py` / `attention.py` | 连续VAE、单遍DF、C4/local16、packed隔离和flow loss |
| `launch.sh` / `train.py` / `monitor.py` | 官方启动器、Trainer及运行回执 |
| `inference.py` | 官方solver与有界KV，逐块自由生成或GT历史诊断 |
| [visualization/README.md](../visualization/README.md) | 完整长短段预览、解码历史、页面与服务 |
| `configs/ego100h.toml` | V0.1最长97帧随机短窗口历史复现，不是新训练默认 |

数据清单是共享存储 `/mnt/lzh/cosmos-EgoWAM/training_manifests/` 的只读输入，不需要联合模型代码。正常段保留全部原帧；单条超预算时采用已批准的90%→50%均匀保留策略，真实时间跨度和有效帧率留在元数据中。

运行环境为 `/home/lzh/miniconda3/envs/cosmos3/bin/python`，`PYTHONPATH=<repo>:<repo>/packages/cosmos3`。训练入口复用 `bash cosmos3_ar_it2v/launch.sh`，新训练需要明确配方授权；不会自动重启已完成实验。推理参数见 `python -m cosmos3_ar_it2v.inference --help`，产物使用新的版本目录，不能覆盖旧结果。

模型设计见 [V0.2设计](../docs/ar_it2v_v0.2/design.md)，工作规则见 [AGENTS.md](../AGENTS.md)。
