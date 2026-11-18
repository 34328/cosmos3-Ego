# Cosmos3 AR IT2V

Cosmos3-Nano 上的纯视频 AR 继续预训练：输入首帧图像与动作段文本，逐块生成视频。迁移 CMD Stage 1 的核心训练方法，保留官方 Cosmos3 的 VAE、文本路径、Packer、Trainer 与 checkpoint；不含 action 模态。

## 当前状态

V0.4 四节点32卡正式训练已于2026-10-09按用户要求停止，最后完成 **2536 步**，最新已保存 checkpoint 为 `iter_000002500`；保留全部权重、日志和评测，不自动恢复或重启。[训练记录](docs/ar_it2v_v0.4/experiment.md)和 [W&B run](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/j29qq2x1)保留来源；W&B 云端终态尚未核实。配方为 GT历史与所有预测块并行监督、LingBot历史噪声、Cosmos3官方flow matching、C4/local32、49920源token预算。统一网页已发布 step500/1000/1500/2500 的完整片段与两种历史对照。

V0.2 完整 segment 基线已完成1500步，V0.3 单目标实验按用户要求停在194步；历史配方和产物保留。

V0.2基线设置（旧配方保留）：

- 一个完整文本 segment 是一个训练样本；正常段使用全部连续原始帧，原始30fps，不随机裁窗口、不跨段。
- 官方动态 Packer 预算75008 token；仅单条超预算段使用批准的90%→50%均匀保留策略，仍超限则明确排除。
- 整段连续 VAE 编码，latent 分块 `[1,4,4,…]`，C4/local16，绝对 RoPE，有界 KV cache。
- 单遍 Diffusion Forcing：首 latent 干净且不计 loss；每个后续 GT 块独立加噪、块内共享 σ，并计算 flow loss。
- 从官方 Nano 初始化、GEN训练/UND冻结；4节点32卡、1500步/save500、lr1e-4→3e-5、weight_decay0.01。

目前自由生成仍有任务偏离与累积误差。GT历史诊断采用匹配GT解码前缀，减少了已观察样本的重影，但这是依赖GT的单块拼图；不能称为自由生成修复。现状和后续问题见 [problems.md](docs/problems.md)。

临时测试先在对话中展示选定视频，不进入正式网页；产物放`outputs/diagnostics/`。用户确认最终版本后才发布到`outputs/visualization/`，沿用统一网页。

## 导航

| 内容 | 入口 |
|---|---|
| 协作规则 | [AGENTS.md](AGENTS.md) |
| 当前设计 / 实验 | [V0.4设计](docs/ar_it2v_v0.4/design.md) / [实验记录](docs/ar_it2v_v0.4/experiment.md) |
| 历史完整段基线 | [V0.2设计](docs/ar_it2v_v0.2/design.md) / [实验记录](docs/ar_it2v_v0.2/experiment.md) |
| 历史单目标实验 | [V0.3设计](docs/ar_it2v_v0.3/design.md) / [实验记录](docs/ar_it2v_v0.3/experiment.md) |
| 当前模型 / 注意力 | [model_v04.py](cosmos3_ar_it2v/model_v04.py) / [attention_v04.py](cosmos3_ar_it2v/attention_v04.py) |
| 完整段数据 / 官方 packing 适配 | [dataset.py](cosmos3_ar_it2v/dataset.py) / [dataloader.py](cosmos3_ar_it2v/dataloader.py) |
| 当前配置 / 启动 | [ego100h_parallel_tf.toml](cosmos3_ar_it2v/configs/ego100h_parallel_tf.toml) / [launch.sh](cosmos3_ar_it2v/launch.sh) |
| 逐块推理 | [inference.py](cosmos3_ar_it2v/inference.py) |
| 统一预览与服务器网页 | [visualization/README.md](visualization/README.md) |
| 官方框架 | [packages/cosmos3/AGENTS.md](packages/cosmos3/AGENTS.md) |
| 历史短窗口实验 | [V0.1设计](docs/ar_it2v_v0.1/design.md) / [实验记录](docs/ar_it2v_v0.1/experiment.md) |

远端 `/mnt/lzh/cosmos-ar-it2v`，分支 `ar-it2v-pretrain`。本地新项目保存本分支源码与文档；数据、权重、运行产物和网页服务在远端。共享数据清单仍位于 `/mnt/lzh/cosmos-EgoWAM/training_manifests/`，作为只读数据源；不复制到项目，不依赖该目录中的联合模型代码。

环境：`/home/lzh/miniconda3/envs/cosmos3/bin/python`，`PYTHONPATH=<repo>:<repo>/packages/cosmos3`。执行前遵守 [AGENTS.md](AGENTS.md)，优先复用官方组件。
