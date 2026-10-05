# Cosmos3 AR IT2V 协作指南

## 项目范围与当前状态

本项目在 Cosmos3-Nano 上迁移 CMD Stage 1 的因果视频训练方法，进行 EgoVerse 图像＋文本条件的纯视频 AR 继续预训练。输入是首帧图像与 segment 文本，输出只有视频；没有 action、state、骨架或联合监督。项目适配放在 `cosmos3_ar_it2v/`，官方框架保留在 `packages/cosmos3/`。

当前基线为 V0.2 完整 segment / 75008 token 配方，已完成 1500 步，最终 checkpoint 为 `iter_000001500`。此前 V0.1 是最长 97 帧的随机短窗口训练，记录仅供历史复现。现阶段推进纯视频诊断与预览；没有新训练、自动恢复或重训授权。新配方需要用户确认后才能启动。

本地项目：`/Users/cnf2026953090/Desktop/rbs-WAM-videogen pretrain`；远端仓库：`/mnt/lzh/cosmos-ar-it2v`；分支：`ar-it2v-pretrain`。本地保存源码和文档，计算、数据、权重与视频服务在远端。主导航见 `README.md`，现状与问题见 `docs/problems.md`，完整实验见 `docs/ar_it2v_v0.2/experiment.md`。

## 模型、数据与训练语义

- 优先复用官方 Cosmos3 的 VAE、文本路径、RoPE、flow 参数化、Packer、Trainer、优化器、调度器、DCP checkpoint 和 W&B。复杂功能先查官方代码及扩展接口；只有已核实的缺口才写最小适配，不另造训练主循环或重复实现。
- 使用官方 Nano 起点，训练 GEN 相关参数、冻结 UND。当前只迁移 CMD Stage 1，不能称为完整复现 CMD，也不混入少步或长视频蒸馏。
- 一个完整文本 segment 是一个样本，保留段边界、文本配对及原始 train/test 划分。正常段连续读取全部原始 30fps 帧，不随机裁窗口、不跨 segment、不改变动作速度。连续编码整段，不能按 AR 块重置 VAE。
- 清单位于共享存储 `/mnt/lzh/cosmos-EgoWAM/training_manifests/`；这是只读数据位置，不是本项目的联合模型依赖。不复制数据，不混入 test。
- 官方 token-budget Packer：`max_samples_per_batch=None`、`max_sequence_length=75008`，packed 样本间注意力隔离。仅单条超预算时采用已批准的 90%→80%→70%→60%→50% 均匀保留帧策略，保持原首尾与时间跨度；50%仍超限则明确排除。记录源索引、真实/对齐长度与 effective_fps，不能静默丢弃或倍速播放。
- VAE 时间对齐保留动作末尾；补帧不计有效监督，支持部分末 AR 块。推理输出去除对齐补帧。
- 单遍 Diffusion Forcing：latent 分块 `[1,4,4,…]`；首 latent 是干净图像条件，不计 loss。每个后续块由 GT latent 独立加噪，块内共享 σ，所有后续块计算 flow loss；训练历史不是前一块的模型预测。σ 为 uniform 经 shift5 后截断至 `[0.02,0.98]`，没有特殊低噪声前缀或前缀 loss 屏蔽。
- 当前注意力 C4/local16 latent 总窗口，当前块内双向、块间因果，历史最多12 latent；无永久首帧 sink。复用官方绝对时间 RoPE、有界 KV cache 和块级去噪/刷新语义，不在新块重置位置。
- 已完成 V0.2 配方：4节点/32卡、HSDP8×4/CP1、seed42、1500步/save500；官方 LambdaCosine warmup100/cycle1500、峰值1e-4/终点3e-5、weight_decay0.01。此处是历史基线，不构成重新启动授权；实际配置和回执以实验文档为准。

## 远端操作与资源

- 所有远端操作只用 MCP SSH Apply Patch，统一账号 `lzh`；禁止本地 SSH/SCP 或 root。首次先查询已知 host，确认仓库、分支、适用 AGENTS、`git log -5` 与 `git status`，查明他人变更，不顺手提交。
- Tdebug1–6 共享存储；不要复制数据/权重，不修改其他仓库或活跃任务使用的源文件、输出。
- GPU 任务前检查 `nvidia-smi` 的利用率、显存和进程；只用空闲资源，不抢占、不杀他人进程。多节点使用内网通信，分别通过各节点 MCP 派发，禁止节点间 SSH。
- 耗时 CPU 阶段先检查 quota、affinity、内存和负载，合理限制 BLAS/OpenMP 线程。启动前核对 claim、真实进程和退出回执，禁止重复派发；失败保留证据，不自动重试或恢复。
- 环境为 `/home/lzh/miniconda3/envs/cosmos3/bin/python`；测试与启动统一 `PYTHONPATH=<repo>:<repo>/packages/cosmos3`。官方规则见 `packages/cosmos3/AGENTS.md`。

## 验证、记录与提交

- 验证只覆盖本次变化的真实路径：数据边界、timestep、mask、loss、历史梯度、KV、VAE及训练生命周期。复用已验收产物；必要检查通过后推进，不重复无关全量实验。CPU 测试显式关闭 CUDA；涉及真实 GPU 行为的变化才做相应短测。
- 正式训练必须 W&B online，启动后用 API 核实本次 run/step/loss/LR，交付真实链接；不输出密钥，不用旧 run 或本地日志代替 API。配置、清单、源提交和资源写入版本实验记录。
- 停止条件以已确认 monitor/配方为准；非有限数值、OOM 等保留完整证据，不因主观画质擅自停止或改配方。
- 输出放独立 `outputs/` 版本目录；数据、权重、checkpoint、视频、缓存和密钥不进 Git。小型必要回执可进 `docs/<version>/evidence/`，临时材料仅放 `tmp/`。
- 独立任务可多 Agent 并行，明确文件负责人；同一文件、配置或资源按依赖顺序处理。提交前审查 diff，分批 commit/push，每次 push 前 `git pull --rebase`，不覆盖他人变更。

## 推理与可视化

- 统一入口 `cosmos3_ar_it2v/visualization/`，先读该目录 README。临时测试放 `outputs/diagnostics/<version>/step<step>/<purpose>/`，保留版本、checkpoint与采样设置，默认不构建网页、不加入已有页面或服务。
- 查看临时结果时只同步用户需要的少量MP4至本地 `tmp/previews/<version>/step<step>/<purpose>/`，直接在对话中预览；不全量下载、不入Git。用户确认最终版本后，才将选定结果发布至 `outputs/visualization/<version>/step<step>/<purpose>/` 并构建/更新正式网页。可移动已验收产物，无须重新推理。
- 默认 train/test 各选几个手部移动明显的拿起、移动、放下动作，长短分别选不同的完整 segment，使用对应完整文本；不从长片裁出短片。页面为 GT RGB／模型 RGB，不显示内部身份、存储路径或联合模型骨架。
- 必须区分 Transformer 的历史 KV 与 VAE 的解码前缀。`generated` 是只给初始图像、后续使用自身预测历史的自由生成，整段预测 latent 连续解码（`decoder_mode=predicted_prefix`）。
- `gt` 是真实历史单块诊断：官方 `get_data_and_condition(..., vision_condition_indexes=None)` 连续编码完整 GT，仅已完成块可写入历史 KV；当前目标块没有自身 GT 条件。显示采用同一 GT 前缀连续解码预测目标块，再按绝对时间裁出 RGB（`decoder_mode=gt_prefix_montage`）。这是 GT 锚定的单块预测拼图，不能当作完整自由生成能力。
- 生成历史的官方首帧优化 `vision_condition_indexes=[[0]]` 不可用于提取完整 GT 历史。不得单独重置 VAE 解码目标块，不得 decode→encode 回填历史。
- 旧 manifest 未记录 decoder_mode 时按旧 `predicted_prefix` 解释，不能把旧 MP4 改名成新解码结果。共享预测末 latent 的额外边界条件在短测中未改善，不设为默认；证据保留，避免重复试验。
- 正式网页服务仅监听远端回环地址，经已有连接转发；只发布已确认页面与选定MP4，不公开manifest、latent、日志、数据或权重。临时测试不加`--extra-page`；临时查看按上一条只同步所需视频。
