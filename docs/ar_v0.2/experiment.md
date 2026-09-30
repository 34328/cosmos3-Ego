# AR V0.2 实验记录

本文只记录**当前方案**的结论和数字。过程日志、逐步验收、半小时跟进、历史表示和审计原文见 [archive/](archive/)，那里的内容按原样保留，不再更新。

产物路径均相对远端 `/mnt/lzh/cosmos-EgoWAM/`。

## 1. 当前方案

| 项 | 值 |
|---|---|
| 动作表示 | `fixed_camera_wrist_local_delta_latent_v1`（57D，补零到 64D） |
| 刚体（相机、双腕） | 块首相机坐标轴下的逐帧增量：`Δp = p_t − p_{t−1}`，`ΔR = R_t R_{t−1}ᵀ`（左乘恢复） |
| 手形 | 当前帧手腕局部坐标 `q = R_wristᵀ(kp − p_wrist)`，PCA15 编码，future 为 `Δz`；换块时 z 直接继承 |
| 手形 codec | 左右手各一个 PCA15，`artifacts/cosmos3_hand_codecs/v3_wrist_local_pca15_train744/` |
| 布局 | C=4、K=8，每块 32 条 action；H=15 个历史 chunk；联合去噪 30 步 |
| loss | `L_video + L_action`，1:1；跟踪有效的全部 future 行都监督（不按 FOV 屏蔽） |
| clip 档位 | (273, 257, 129, 65, 33)，frame_stride=2，speed_factor=0.5 |
| 学习率 | base 2e-5；`action2llm / llm2action / action_modality_embed / action_state_embed / vision_condition_embed` 各 5×；warmup 100，cosine |
| 并行 | 双节点 HSDP，shard 8 × replicate 2，NCCL 走内网 TCP |

设计与公式以 [design.md](design.md) 为准。

## 2. 数据

从 EgoVerse Mecka 100h 候选中筛选，train 与 heldout 按 `(user, scene)` 隔离。

| split | episodes | segments | 小时 |
|---|---:|---:|---:|
| train | 744 | 4409 | 17.34 |
| heldout | 119 | 642 | 2.73 |

- 清单：`outputs/data_expansion_20260928/{episodes,segments,tasks}.csv`。
- 当前统计与窗口清单：`outputs/data_expansion_20260928/prepared_fixed_camera_wrist_local_delta_latent_v1_t273/`（state 286,432 行，future 9,165,824 行）。
- 273 帧 heldout 评测窗口：1,557 个，覆盖 98 个 episode（`outputs/maintenance/formal_t273_20260930/eval_windows_273.json`）。

## 3. 手形 codec：PCA15

744 个 train episode 拟合，119 个 heldout episode 验收（每侧约 28.8 万帧，按关节计）。

| | 解释方差 | mean | P95 |
|---|---:|---:|---:|
| 右手 | 97.17% | 2.95 mm | 6.53 mm |
| 左手 | 97.67% | 2.79 mm | 6.32 mm |

门槛为 mean ≤ 5 mm、P95 ≤ 15 mm，左右手均通过。换块不重编码，100 块长链关键点最大误差 1.8e-7 m。此前尝试的 MLP AE（相机轴手形）未通过 32 次换轴重编码测试，已放弃，记录见归档。

## 4. 正式训练：1000 步

| 项 | 值 |
|---|---|
| 运行目录 | `outputs/joint_video_hand_pose/ar_v0_2/formal_fixed_camera_t273_20260930T011945/` |
| commit | `b3585fb` |
| W&B | [bytyfmqa](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/bytyfmqa) |
| 节点 | Tdebug2 + Tdebug6，16 × H800 |
| 时间 | 2026-09-30 01:19 启动，08:37 结束，两节点 exit 0 |
| checkpoint | 第 500、1000 步 |
| 每步 | global batch 约 38–57 个 clip，同步训练段约 25 s |
| 显存 | 峰值 allocated 64.4 / reserved 75.5 GiB，步首驻留稳定在 15.2 GiB |

loss（区间均值）：

| 步 | video | action |
|---|---:|---:|
| 1–10 | 0.287 | 0.305 |
| 90–110 | 0.139 | 0.114 |
| 490–510 | 0.133 | 0.097 |
| 980–1000 | 0.131 | 0.092 |

video loss 在约 100 步后基本不再下降；action loss 持续缓慢下降。

## 5. 评测结果（第 1000 步）

**范围说明**：以下只覆盖 8 个 heldout 窗口、136 个块，每窗口 17 块。全量 1,557 个窗口的评测因 codec 设备 bug 中断，尚未完成。结论只看趋势。

块末误差，格式为 `模型 / 不动基线`，越小越好：

| 模式 | 相机 mm | 右腕 mm | 左腕 mm | 右腕角度 | 手形 mm | 视频 PSNR |
|---|---:|---:|---:|---:|---:|---:|
| oracle | **16 / 26** | **36 / 45** | **40 / 60** | **16° / 19°** | 8.0 | 34.0 |
| gt | 31 / 26 | 57 / 45 | 65 / 60 | 25° / 19° | 8.4 | 18.4 |
| generated | 220 | 243 | 282 | 53° | 18.9 | 12.6 |

- **oracle**（当前块给真实视频）：相机和双腕都明显优于不动基线，从 500 步到 1000 步仍在改善。说明动作表示可以学会。
- **gt**（当前块用模型自己生成的视频）：比不动基线还差，500 步和 1000 步几乎没区别。
- **generated**（全自回归）：误差逐块累积，第 17 块以后右腕误差达 340–600 mm。

**瓶颈在视频生成**。单窗口像素分析显示，块内生成视频的运动幅度只有真实的约 57%，块末误差与"块首帧原样定格"几乎相同（22.5 对 23.0）；GT 模式的块边界跳变就是每块定格、下一块又被真实帧拉回造成的。

评测产物：`outputs/maintenance/claude_diag_20260930/`（`metrics_step{500,1000}_{gt,oracle,generated}.json`）。对比视频在本地 `eval_videos/ar_v0.2_step1000_oracle/`。

## 6. 负结果

### 6.1 平移改为相对块首帧的累计位移：已终止

动机：正式训练中平移字段 loss 几乎不降，且抽样发现逐帧位移里高频成分占比较高（头 45%、右腕 29%、左腕 35%），怀疑差分放大了跟踪噪声。

做法：只把 future 平移行从 `p_t − p_{t−1}` 改为 `p_t − p_b`，其余完全相同，1000 步。

同 8 个窗口、第 1000 步，手腕位置二阶差分 RMS（mm/帧²）：

| 模式 | 逐帧增量 | 累计位移 | GT | 块末误差 增量/位移（不动基线） |
|---|---:|---:|---:|---:|
| oracle | 2.53 | **14.34** | 2.28 | 37.9 / 37.5（52.2） |
| gt | 1.97 | **10.54** | 2.28 | 61.2 / 62.9（52.2） |

累计位移的抖动是逐帧增量的 5–6 倍，误差没有改善。逐帧增量的预测抖动与 GT 相当，"差分放大噪声"的假设不成立。代码、分支、统计和训练产物已全部删除，不进入方案。

### 6.2 相机轴手形 + MLP AE：已放弃

手形在块首相机轴下表示时，AE 必须同时编码手的朝向；换块需要解码→旋转→重编码，32 次后漂移 47–117 mm，未过门槛。改为当前帧手腕局部坐标后问题消失（见第 3 节）。

## 7. 已知问题

| 问题 | 状态 |
|---|---|
| 视频生成运动不足、质量差 | 当前主要瓶颈，待诊断 |
| 推理未用 CFG，shift=5 | 官方 image2video 默认 guidance 6.0、shift 10、35 步，与我们不同，待验证 |
| `speed_factor=0.5` 使时间标签为 7.5fps，实际视频 15fps | 待单独对照 |
| 全量评测因 codec 设备 bug 中断 | 已在对照分支修复（已删），需移植到主线后重跑 |
| 满窗口（H=15）与淘汰旧块 | 训练已通过 273 档覆盖；长 rollout 时间 RoPE 外推尚未单独评估 |
| 推理延迟 | 按用户决定暂缓 |

## 8. 历史训练（旧表示，仅作参考）

2026-09-28 用旧动作表示（块首 state、旧 wrist-local AE、C=1…4、`L_video + 0.7 L_action`）跑过 1200 步，单节点 FSDP8，W&B 误设 disabled，分项 loss 历史丢失。与当前方案不可比。详见归档。
