# AR V0.3.0 实验记录

## 1. 范围与版本

本轮依据本地同步的 `docs/ar_v0.3/design.md`，仅改变学习率/调度、单遍 diffusion forcing、57D 手腕通道权重。V0.2 文件保持不动；57D/PCA15、C4/K8/H15、混合 clip 档位、60K token 预算、数据/normalizer/codec、联合推理 30 步沿用。

- 核心实现 commit：`2cf54ed`，已 push 到 `origin/ar-video-action`。
- 注册名：`rbs_wam_ar_v0_3_diffusion_forcing_wrist_weight_v1`。
- 新训练入口：`cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_3.sh`；官方 Cosmos CLI / Trainer / optimizer / scheduler / checkpoint / W&B 保留。
- 同步设计 SHA256：`68b2e84afcb6da32f1dc550aae09946cfe7a28e83b85deab05ec3f1d1e87481b`。腕部加权占比按修正后的 58%，即 54/93。

## 2. 学习率回执

全模型峰值 `1e-4`，五个 action/condition 模块倍率均为 1。官方 `LambdaCosine`：warmup 100，cycle_lengths=[3000]，f_start=0，f_max=1，f_min=0.3。正式 max_iter=3000、save_iter=500。官方周期长度包含 warmup；CLI 最终配置在 optimizer 初始化前断言并打印/保存回执。

| step | 理论 LR |
|---:|---:|
| 0 | 0 |
| 100 | 1e-4 |
| 1500 | 6.689486180048961e-5 |
| 3000 | 3e-5 |

## 3. CPU / GPU 验证

回执目录：`outputs/maintenance/ar_v03_implementation_20261002/`。测试均在远端 `lzh`、`cosmos3` 环境（torch 2.10.0+cu128）执行。GPU 测试前确认 Tdebug6 无 compute 进程，再分别使用空闲 GPU0/1/2。

- CPU 主套件：62 passed、1 CUDA skipped（`all_cpu.log`）；eval metadata 新补 1 项后专项 22 passed；短测 callback 专项 8 passed（`smoke_monitor_cpu.log`）。
- 全 1 通道权重 loss / 梯度 / metrics 与 V0.2 精确一致；非均匀权重的 mask/分母/字段未加权日志通过。loss CUDA 专项 19 passed（包含 CPU 项及 1 CUDA 项，`loss_gpu.log`）。
- Attention CPU 7 项、CUDA 5 项通过：单遍 mask 与 V0.2 clean mask 等价；块因果、H15、样本隔离、U/S 条件隔离及 FP32/BF16 梯度。
- 官方完整小网络 GPU 2 项通过（`network_gpu.log`）：真实 training_step 只调用一次 net，所有梯度有限；8 个字段日志为未加权值；无 clean KV 保留。
- Refresh CPU 专项 22 项、官方小网络/KVCache CUDA 2 项通过：sigma_small=0 与原始输出/缓存逐位一致；非零 refresh 同时作用于历史 V/A，U/S 和当前预测不变，后续块受影响；C4 尾块通过。
- `py_compile`、启动脚本 `bash -n` 通过。环境没有 ruff，不宣称执行过 lint。

用户补充的 noisy 块历史梯度：对完全相同的 3 次完整网络前向分别反传本块 loss、后续块 loss、两者之和，避免编译 attention 的 donated-buffer 对 retain_graph 限制。块 k 的视频和 action 输入收到两条路径的非零梯度，合并梯度等于各路径之和；未来块梯度严格为 0。

| 完整网络 FP32 输入 | 自身块 loss 梯度 L2 | 后续块 loss 梯度 L2 |
|---|---:|---:|
| 块 k 视频 | 0.039788905531167984 | 0.00028181084780953825 |
| 块 k action | 0.06347619742155075 | 0.0011661481112241745 |

两层 attention CUDA 对照也在 FP32/BF16 验证了自身/后续/合并梯度，以及 U/S 不通过历史读取、块17 H15淘汰和跨样本零梯度。

## 4. 单节点 20 步与 V0.2 两遍对照

状态：V0.3 短测已启动，结果待回填。正式训练尚未启动，须先向用户交付短测结果并获确认。

- 节点 Tdebug4：8×H800（约79.6 GiB/卡），CPU quota 120 核、affinity 200 核，启动检查 MemAvailable 约1.37 TiB、GPU 无 compute 进程。GPU 任务逐组串行；不抢占其他进程。
- 独立比较目录：`outputs/maintenance/ar_v03_short_compare_20261001T163632/`（目录采用服务器 UTC 时间；日志采用服务器本地时间）。
- V0.3 与 V0.2 各20步，shard8/replicate1/CP1、seed42、同一官方 Nano 起点。V0.2 使用保留的 `ar_v0_2_video_lr1e4.toml`，只在 CLI 覆盖 cycle3000/f_min0.3/max_iter20/save20/replicate1 与短测 job 名称。
- 两组最终 dataloader_train/val、optimizer、scheduler、checkpoint 配置完全相同。实际窗口身份逐 rank/step hash 验证后才比较耗时。
- 仅此短测 CLI 和环境使用 W&B disabled；正式 V0.3 TOML/注册保持 online。
- 新 `ARV03SmokeMonitor` 只读诊断：计数 training_step 前向（排除 backward checkpoint 重算），记录7个模块组的 raw finite/nonzero 梯度和官方 preclip norm、窗口身份、GPU峰值与 step 时间。
- step 时间包含模型准备、forward/backward、optimizer/scheduler/zero_grad；不含 dataloader/H2D/checkpoint。梯度诊断开销单独记录。报告首步/JIT与稳态均值，不能用小网络耗时替代正式 Nano 的实测。
- launcher：`cosmos3_joint_video_hand_pose/scripts/ar_v03_short_compare.sh`，分别参数 `v03` / `v02`，两次共用 `COMPARE_ROOT`。会拒绝已有 started.txt 的重复任务及有 compute 进程的节点。

## 5. 正式训练与评测待办

用户确认短测后，重新检查实时资源，选两台空闲节点 HSDP 8×2，跨节点 NCCL 内网 TCP；在线 W&B 启动前核验最终配置/环境，启动后通过 API 核实 step/loss 已上传并记录当前 run 链接。沿用 `FormalTrainingMonitor` 与 `StopPolicy`，不改变 OOM 停止规则/预算，不擅自重启。

step 500/1000/2000/3000 每次 heldout8 gt/generated（sigma_small=.02）+固定训练16 gt。全部32窗口、冻结清单hash、320×180整体光流余弦，以及 V0.2 step1000 / lr1e-4 同表方案见 `evaluation_plan.md`。未运行的训练/评测不填虚构结果。
