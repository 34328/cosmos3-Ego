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
| 全量评测因 codec 设备 bug 中断 | 主线修复 `b513d67`：每窗口推理独立 codec 副本；43 项 CPU 回归通过，四进程各连续两窗口 GPU 验证通过。尚未重跑全量评测 |
| 满窗口（H=15）与淘汰旧块 | 训练已通过 273 档覆盖；长 rollout 时间 RoPE 外推尚未单独评估 |
| 推理延迟 | 按用户决定暂缓 |

## 8. 历史训练（旧表示，仅作参考）

2026-09-28 用旧动作表示（块首 state、旧 wrist-local AE、C=1…4、`L_video + 0.7 L_action`）跑过 1200 步，单节点 FSDP8，W&B 误设 disabled，分项 loss 历史丢失。与当前方案不可比。详见归档。

## 9. 2026-09-30 视频效果排查（推理部分已完成，未训练）

- 独立输出：`outputs/maintenance/codex_ablation_20260930T191612/`。
- 固定正式 `formal_fixed_camera_t273_20260930T011945` 的 iter1000、`claude_diag_20260930` 实际完成的同 8 个 heldout 窗口，seed 42、C4、AR 联合 30 步；冻结清单及 hash 在新目录 `manifest.json`。
- 顺序：时间位置／缓存重算／VAE／边界数值核对 → 视频 shift 5/10 → 固定较好 shift 后视频 CFG 1/3/6 → 官方 Nano image2video（guidance 6、shift 10、35 步）。训练对照等待用户另行确认。
- 只改视频 schedule／guidance，action shift 固定 5、guidance 固定 1。CFG 的空文本使用官方训练 tokenizer，同步维护独立无文本历史 KV。
- shift 选择标准预先固定：8 窗口 gt 模式左右腕块末误差均值较小者；相同则按视频 PSNR。所有相机／手形／视频得失一起报告。
- 每组报告 8 窗口、136 个 chunk，单列 chunk>=17；可视化固定窗口 0/1/2（穿吊牌、搅拌蜡液、扣衬衫纽扣），不按效果挑选。原生 image2video 没有项目 57D 动作输出，动作指标标为不适用。
- 新报告视频 PSNR 先汇总有效 RGB（去掉底部 padding）的像素 MSE 再转 dB，不能与旧报告的逐块 dB 算术均值直接比较。边界 MAE 比较同一源时刻的预测末帧与下一块 U，不将像素变化量解释为运动速度。
- 初始检查：六节点 GPU 空闲；Tdebug4 CPU 负载较高。Tdebug1 四卡各连续核对两窗口，Tdebug2 八卡完成 shift10，Tdebug3/5 各八卡完成 CFG3/6；释放后的 Tdebug1 完成官方 Nano。CPU 像素汇总先验证一个窗口，再八进程、每进程两线程并行，复用首窗口结果，汇总耗时78.15秒。

### 9.1 数值核对

| 项目 | 实测 |
|---|---|
| 8 窗口时间位置 | 首块视频 latent 源帧 8/16/24/32，与各组 action 末行 RoPE 差均为 0；U/S 均在源帧 0，future action 为 1…32 |
| 缓存与完整前缀重算 | 8 窗口×17块×30步×video/action=8,160 项，max_abs=0、relative_L2=0，含 k17 首次淘汰；逐步使用相同输入比较 |
| 8 窗口 VAE PSNR（逐窗口平均） | 整段 33.653 dB；分块 33.971 dB；差值 +0.318 dB，范围 +0.213…+0.470 dB |
| 同时刻块边界 MAE（0…255） | gt 预测 25.099；oracle 真实视频经分块 VAE 重建 2.594；generated 1.915，但其首条件对 GT 的 MAE 已达 42.462，平滑不等于准确 |
| 修复后复现 | 首批 4 窗口的 predicted_action 和 generated_rgb 与原诊断逐元素完全一致；四个进程均成功编码并推理第二窗口，codec 设备错误未复现 |

结论：当前证据未指向时间错位、缓存数值错误或 VAE 分块重建明显劣化；较大的边界偏差来自生成结果与真实条件的差距。详见 `step2_initial_report.json`、`step2_cache_final.json` 及逐项 JSONL。对照代码 `8ba69fc`，85 项 CPU 回归通过；补充独立缓存验证后 guidance 测试 7/7 通过。

### 9.2 shift 与视频 CFG（已完成）

每行同 8 窗口、136 块。单位：位置/手形 mm；PSNR dB；边界 MAE 为 0…255。手形误差在当前腕局部坐标、对原始 GT 关键点计算。generated 位置已包含累积漂移。

| 历史 / video shift / CFG | 相机块末 | 右/左腕块末 | 右/左手形 | 视频 PSNR | 边界 MAE |
|---|---:|---:|---:|---:|---:|
| gt / 5 / 1 | 31.15 | 57.33 / 65.03 | 8.42 / 8.36 | 16.93 | 25.10 |
| gt / 10 / 1 | 30.99 | 58.00 / 64.40 | 8.31 / 8.30 | 16.88 | 25.49 |
| gt / 5 / 3 | 31.50 | 57.30 / 65.71 | 8.29 / 8.15 | 17.03 | 24.93 |
| gt / 5 / 6 | 31.81 | 61.11 / 68.38 | 8.16 / 8.11 | 17.02 | 25.06 |
| generated / 5 / 1 | 219.97 | 243.13 / 281.79 | 18.93 / 18.08 | 12.22 | 1.91 |
| generated / 10 / 1 | 156.78 | 196.18 / 254.25 | 18.81 / 18.89 | 12.14 | 1.96 |
| generated / 5 / 3 | 125.55 | 172.97 / 220.94 | 18.11 / 17.70 | 12.17 | 1.88 |
| generated / 5 / 6 | 93.93 | 151.20 / 189.81 | 20.28 / 18.28 | 11.95 | 2.00 |

- 不动基线每块从真实起点出发，相机/右腕/左腕块末误差 25.77/44.71/59.67 mm；不等同于自由自回归的无重置基线。
- 预设 shift 选择指标（gt 双腕块末平均）为 61.1776 vs 61.2039 mm，近乎持平；后续固定 5。shift10 在本批自回归漂移上较好，但像素 PSNR 无改善，不宣称显著优劣。
- CFG 在本批样本减轻累积漂移，但 GT 单块腕误差没有改善；CFG6 的自回归 PSNR、右手形更差。单靠调 shift/CFG 未解决问题。
- 每组 `summary.json` 保存逐窗口、逐块和 `chunk17plus` 指标。第17块自回归右/左腕：基线341.19/436.31；shift10为251.11/378.04；CFG3为255.27/340.95；CFG6为205.38/286.03 mm。仅8个第17块，不作大样本结论。
- 实验目录组名：`baseline_{gt,generated,oracle}`、`shift10_cfg1_{gt,generated}`、`cfg{3,6}_{gt,generated}`。各组固定前三窗口有 `window00…02.mp4`。
- 官方 Nano 对照见下一节。

### 9.3 官方 Nano 原生 image2video（已完成）

使用未修改的 `cosmos_framework.scripts.inference` 和注册 `Cosmos3-Nano.yaml`，官方预训练权重 `/mnt/checkpoints/Cosmos3-Nano`，同样8个首帧/文本/seed42，273帧、15fps、guidance6、shift10、35步、默认 UniPC 与 diffusion cache；不改写生成循环。关闭一次性冷启动的 torch compile、关闭 guardrail/人脸模糊后处理以记录原始生成像素；没有触发安全拒绝。asset-only overrides 将 VAE 后端、权重和 VLM JSON 指向本地，保留两次资产配置失败日志（云存储凭据路径、官方 loader 改 cwd 后相对 JSON 路径失效）。成功目录 `native_nano_absolute_assets/`，8项官方 `sample_outputs.json` 均为 success。

统一按官方等比例缩放与中心裁切到832×480；GT使用同一源帧，去掉初始条件，合计8×272个future采样帧，先汇总MSE再转dB。读取官方 `output.safetensors`，不从有损MP4反算指标。表中的边界指标为源帧32k到32k+2的相邻帧MAE，与9.1/9.2同一时刻条件重置MAE不同，不能混用。

| 方法 | PSNR | 边界相邻帧MAE | 第17块PSNR |
|---|---:|---:|---:|
| AR gt shift5 CFG1 | 17.13 | 24.68 | 17.67 |
| AR generated shift5 CFG1 | 12.27 | 8.16 | 12.10 |
| AR generated shift10 CFG1 | 12.20 | 7.89 | 11.86 |
| AR generated shift5 CFG3 | 12.22 | 8.02 | 11.49 |
| AR generated shift5 CFG6 | 12.01 | 6.11 | 11.27 |
| 官方 Nano image2video | 11.84 | 5.76 | 11.28 |
| 全程保持首帧不动 | 12.71 | 0 | 12.29 |
| GT | — | 6.50 | — |

原生 Nano 不输出项目57D动作，相机/双腕/手形误差均不适用；它没有AR条件重置，因此同一时刻重置误差也不适用，不填伪造的0。每组每窗口/块的共同像素网格指标和完整表存于 `native_comparison/{metrics00…07,common_grid}.json`；固定前三窗口对比视频为 `window00…02.mp4`，左GT连续30fps、右预测15fps按真实时间显示，无伪造手形投影。

结论：官方原生也不逐帧复现唯一GT未来，静止首帧的PSNR反而更高，不能用PSNR单独判断视觉质量或任务动作正确性。原生与AR同时存在权重、分辨率、时间标签（15 vs 7.5fps）、整段/分块生成、采样器与步数等差异，这是原生能力参照，不能据此单独归因于某项改动。已测证据排除了这8窗口的时间对齐/缓存错误及明显VAE分块劣化；shift/CFG不能解决GT单块腕误差。第5步训练对照等待用户确认，未启动。

本地回放：`http://127.0.0.1:54712/ablation_20260930/`（需本地媒体服务运行）；本地文件 `eval_videos/ar_v0.2/ablation_20260930/`。所有对照保留相同8窗口和固定3个展示样本，不替换失败或不好看的样本。

### 9.4 pred_history 归因对照（2026-09-30，未训练）

复用9节冻结8窗口、正式step1000、seed42、shift5/CFG1、C4联合30步；gt/generated直接读取原 `baseline_{gt,generated}/summary.json`，未重跑。新目录 `outputs/maintenance/codex_predhist_20260930T222604/`，运行代码 `7648482`；Tdebug1空闲8卡各处理1窗口。原实现语义符合要求，无代码修改：每块U/S取GT，历史future V/A保留预测后写入KV，历史U/S保留当时GT。8窗口运行时检查：全部17块U/S与gt一致、首块action/RGB与gt一致，最大差异均为0；checkpoint、codec、两套统计hash一致。

| 模式 | 相机块末mm | 右/左腕块末mm | 右/左手形mm | PSNR dB |
|---|---:|---:|---:|---:|
| gt | 31.15 | 57.33 / 65.03 | 8.42 / 8.36 | 16.93 |
| pred_history | 95.23 | 116.36 / 132.35 | 13.29 / 11.50 | 12.34 |
| generated | 219.97 | 243.13 / 281.79 | 18.93 / 18.08 | 12.22 |

不动基线（每块GT起点）：相机25.77、右腕44.71、左腕59.67 mm；不等同于无重置自由rollout基线。全部指标沿用原始GT关键点、640×360有效像素；PSNR先平均像素MSE再转dB。

按 `(pred_history−gt)/(generated−gt)` 计算差距比例：相机33.9%、右腕31.8%、左腕31.1%，右/左手形46.3%/32.3%；视频在MSE域为95.9%。因此**视频接近generated，历史是这项退化的主要来源；动作位置介于两者，GT块首条件消除了约66–69%的额外误差，但历史误差仍明显。** 不能只增强块首条件而忽略历史；后续应验证历史（尤其视频历史）增强，动作块首条件鲁棒性也需考虑。上述比例是本批8窗口、单seed的结果差距，不是两种因素可加的独立因果贡献；也未区分U与S各自作用，未证明某种增强配方一定有效。

逐块曲线：第1块三组完全重合（右腕46.52 mm、PSNR14.99）；到第3块pred_history视频已降到12.24 dB，与generated的12.30接近，之后约11.6–12.7波动；gt保持约15–19 dB。pred_history右腕升高后波动，并非随块数单调增长；generated呈明显累积增长。第17块右腕gt/pred_history/generated为43.95/85.94/341.19 mm，PSNR为17.43/12.23/12.06 dB。

产物：新目录 `comparison.json`、`runtime_validation.json`、`curves.csv/json/png/svg`（1…17块、每块8窗口）；完整逐窗口指标在 `inference_pred_history/summary.json`。网页已加入相同前三窗口的pred_history视频与曲线，位于本地 `eval_videos/ar_v0.2/ablation_20260930/pred_history/`。本次页面改为内嵌统计数据，可直接打开本地 `index.html`，不依赖HTTP读取JSON；本会话网络权限不允许重新监听54712端口。未启动任何训练。

## 10. 过拟合与历史视频加噪并行对照（2026-10-01）

用户授权两组同时启动；本节区分计划、已通过测试和实际运行结果，不把未完成评测写成结论。

| 组 | 数据和目标 | 步数/保存 | 节点计划 |
|---|---|---|---|
| overfit16_clean_history | train split 固定16个273帧窗口，16个不同episode；不加历史噪声 | 300步，每100步保存并验收 | Tdebug1 + Tdebug3，16卡 |
| history_video_noise_p50_s020 | 完整原train split；每sample 50%概率扰动历史视频，每历史chunk独立σ~Uniform(0,0.2) | 1000步，每500步保存 | Tdebug2 + Tdebug6，16卡 |

两组从同一官方Nano初始化，复用官方 `launch_ar_v0_2.sh` → Cosmos Trainer；HSDP shard8/replicate2，C4/K8、历史15chunk、60K预算、PCA15和两套57D统计不变。加噪组除新增历史增强和job身份外，配置逐项对照原正式配方一致：LR2e-5、指定五组×5、warmup100/cycle1000/f_min0.1、联合loss系数1、确定性cuDNN、clip tiers=(273,257,129,65,33)。过拟合组仅另改窗口白名单、max_iter=300、save_iter=100、num_workers=1；避免16窗口在16×3个iterable worker间出现空分片，其学习率曲线保留原cycle1000。原始全量audit和normalizer绑定hash不改，不用小集合重新拟合统计。

增强公式 `(1−σ)·V + σ·ε`，ε为标准高斯。仅在teacher-forcing条件前向的独立video副本施加；U、S、历史action输入及GT监督目标不改。对应video timestep=σ×官方训练时间尺度；官方 `mse_loss_indexes` 在该条件前向中只用于timestep路由，该副本不参与loss计算。独立seed由train seed/iteration/rank派生，不消耗原目标噪声RNG；相同历史chunk供后续query读取同一份噪声。σ=0返回原pack，走原数值路径。

准备/测试目录 `outputs/maintenance/training_diagnostics_20261001/`。固定训练清单 `cosmos3_joint_video_hand_pose/configs/overfit16_windows_20261001.json`，SHA256 `7e088380b461fce4ea34ec6aeea141d445d1aff6c3acf81f2cbfb401a6bb9ee4`；train/heldout不交叉，保留原8窗口hash `8ac5c99d0a80f963b03858dd9260e12aa3cccead0e0bf4475427bd863ec1f03d`。完整seed、episode列表、原审计hash在该目录manifest.json。

已通过回归：CPU主组123通过/1项GPU测试跳过，补充组27通过/1项GPU测试跳过，GPU 7通过。覆盖C=1…4与尾块、σ=0官方网络输出逐位一致、σ>0仅直接修改历史V及其timestep、目标/动作/U不变、概率抽样与独立RNG，以及配方全量等价和模型条件/目标hook。CPU组合配置检查及16个独立episode/精确窗口核验通过；GPU数值测试使用小骨干和官方网络/attention模块，不冒充完整Nano训练生命周期验收。

评测：过拟合每100步在固定16训练窗口检查gt/oracle，并对照“不动”与原step1000；最终两组模型在同一8个heldout窗口、同一seed、shift5/CFG1、C4联合30步上扫历史σ={0,0.05,0.1,0.2}和gt/pred_history/generated。原step1000也做同样网格，已有σ=0结果经hash核对复用。推理噪声只在完成chunk写入历史KV时施加，保存/显示的预测及下一块U/S构造仍用未加噪的结果；不额外对当前目标去噪加噪。逐块曲线报告相机/左右腕、手形、PSNR、边界跳变，chunk17单列；主指标使用原始GT。新输出独立，不覆盖9节诊断。W&B必须online，并以API核实真实step/loss后记录链接。训练尚未启动。
