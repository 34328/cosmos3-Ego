# Cosmos3 AR IT2V 协作指南

## 项目与当前授权

本分支 `ar-it2v-pretrain` 在 Cosmos3-Nano 上迁移 CMD Stage 1 的核心方法，进行 EgoVerse 图像＋文本条件的纯视频 AR 继续预训练。项目扩展统一放在 `cosmos3_ar_it2v/`；设计和实验记录放在 `docs/ar_it2v_v0.1/`。交流与文档默认中文，代码标识符保留英文。

用户已停止最长97帧的随机短窗口实验。本轮只授权完整segment接入、官方token-budget packing、必要CPU/GPU短测和记录；新的正式训练配方须由用户确认后启动，不自动恢复旧run或自动重训。具体配方以本分支设计、配置和实际运行回执为准。

## 核心约定

- 输入只有首帧图像和文本；生成模态只有视频。没有 action、state、audio 或 LiDAR 输入、token、投影头和监督。
- 从官方 Cosmos3-Nano 权重起步。复用其视频 VAE、文本路径、位置编码及 flow 参数化；训练 GEN 相关参数，冻结 UND 路径。
- 数据使用 `training_manifests` 原始 train/test 划分与动作段文本。实际清单位于 `/mnt/lzh/cosmos-EgoWAM/training_manifests/`；不复制共享原始数据，不混入 test。
- 原始视频为 30fps：连续读取源帧，`frame_stride=1`，不抽帧、不倍速。VAE 的原生时间压缩不属于数据抽帧。
- 连续编码完整segment，不在每个AR块重置VAE；原始边界和文本配对不变，不随机裁短窗口、不跨segment。末尾VAE对齐必须保留真实长度、排除补帧计数，支持部分末AR块。旧97/81/65/49/33/17随机窗口配置仅保留作历史复现，不再作为新训练默认。
- 单遍 Diffusion Forcing，latent 分块为 `[1,4,4,…]`。首个 latent 是干净图像条件，其余块内共享噪声、块间独立；首帧从 loss 分子和分母排除，所有未来块参与 flow loss。
- 当前注意力为 C4、local16 latent 总窗口，当前块内双向、块间因果，历史最多 12 latent；不额外保留永久首帧 sink。各 packed 样本必须隔离。
- σ 采样为 uniform 经 shift5 后截断到 `[0.02,0.98]`，首帧为 0；不引入低噪声前缀或前缀 loss 屏蔽。推理按整个块去噪、按整个块刷新 KV，保留绝对时间位置。
- 当前只迁移 Stage 1，不把少步蒸馏或长视频蒸馏混入同一次实验。

## 远端入口与资源

- 开发、测试、训练与推理均在远端进行。本分支工作目录为 `/mnt/lzh/cosmos-ar-it2v`；本地仅保留协作资料。
- 远端操作只能使用 MCP SSH Apply Patch 插件，统一账号 `lzh`，禁止自行改用本地 SSH、SCP 或 root。先调用 `listKnownHosts`，以当前工具 schema 的 `hostAlias` 等真实字段执行命令。
- 获准节点为 Tdebug1–6。首次工作先读适用 AGENTS、确认路径、`git log -5`、`git status` 与环境；不能用本地状态代替远端状态。
- 六节点共享存储。不要并发修改同一文件或复制数据/权重；保持旧仓库与其他 worktree 独立，不改动其他正在运行任务的源文件或输出。
- 启动 GPU 任务前检查 `nvidia-smi` 的显存、利用率与进程。忙则换空闲 GPU/节点，不杀他人进程、不抢占资源。多节点通信使用内网；未验证节点间 SSH 时，分别通过 MCP 启动两端。
- 耗时 CPU 阶段先检查 CPU quota、affinity、空闲内存与负载。独立工作合理并行，每进程限制 BLAS/OpenMP 线程；复用已验证产物，禁止不必要的全量重算。
- 启动前检查 launch claim、进程、输出及退出回执，避免重复任务。失败保留完整证据；没有明确恢复方案和授权时不自动重启。

## 训练、验证与记录

- 复用官方 Cosmos3 CLI、Trainer、优化器、调度器、DCP checkpoint 和 W&B。扩展点足够时不另写训练主循环，不全局劫持官方日志。
- 运行环境：`/home/lzh/miniconda3/envs/cosmos3/bin/python`；测试与启动脚本都使用 `PYTHONPATH=<repo>:<repo>/packages/cosmos3`。官方框架的额外导航与规则见 `packages/cosmos3/AGENTS.md`。
- 验证聚焦真实 noising、timestep、mask、loss、历史梯度、cache 等价性及正式训练生命周期。必要检查通过后推进工作，不重复已通过的同一套测试。
- CPU 回归显式关闭 CUDA；原生配置的运行期验证可能初始化 CUDA，放在实际 GPU 短测中验证。
- 正式训练必须 W&B online；启动前核实最终配置和环境覆盖，启动后通过 API 确认真实 run/step/loss 已上传，交付可点击链接并记入实验文档。不得输出密钥，不用旧 run 或本地日志代替 API 核实。
- run 名称包含版本与实验方向，日期不能成为主要标识。记录 source/config/manifest hash、资源、吞吐、loss、裁剪前梯度与显存。
- 梯度裁剪是训练算法的一部分；频繁触发本身不等于发散。具体停止条件以本分支 monitor 和已记录配方为准；保留非有限数值、OOM、持续 loss 爆炸及显存异常的失败证据，不因主观画质判断擅自停训。
- 输出写入独立 `outputs/` 子目录，checkpoint、视频、NPZ、缓存不入源码 Git。必要小型回执与实验结论写入版本文档；临时材料放 `tmp/`。

## 可视化

- 纯视频预览入口为 `cosmos3_ar_it2v/visualization/`，先读其中 README，复用现有 sampler、固定清单和页面。产物按 `outputs/visualization/<version>/step<step>/<purpose>/` 隔离；保留原始30fps，GT／生成RGB双栏，不引入联合任务骨架或旧17块规则。
- 默认 train/test 各三个按真实GT选择的明显操作窗口；短片为同次长生成的动作区间裁剪。超过训练最大97帧的长生成必须注明外推长度。真实checkpoint、seed、去噪次数、CFG与历史刷新噪声均展示。
- 网页服务仅监听服务器回环地址，经已有远程连接转发，不自动下载视频副本。只发布页面及选定MP4，不公开日志、样本身份、清单、权重或训练目录。

## 协作与提交

无依赖工作优先多 Agent 并行，明确文件写入负责人；涉及同一配置、代码或运行资源的步骤按依赖顺序推进。提交前检查 diff，不能顺手提交他人变更；分批 commit/push，每次 push 前先 `git pull --rebase`。有冲突先核对来源，不覆盖正在训练使用的版本。旧实验历史由其他分支保留，不在本纯视频分支重新引入其目录或路线约束。
