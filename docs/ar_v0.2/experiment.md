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

评测：过拟合每100步在固定16训练窗口检查gt/oracle，并对照“不动”与原step1000；最终两组模型在同一8个heldout窗口、同一seed、shift5/CFG1、C4联合30步上扫历史σ={0,0.05,0.1,0.2}和gt/pred_history/generated。原step1000也做同样网格，已有σ=0结果经hash核对复用。推理噪声只在完成chunk写入历史KV时施加，保存/显示的预测及下一块U/S构造仍用未加噪的结果；不额外对当前目标去噪加噪。逐块曲线报告相机/左右腕、手形、PSNR、边界跳变，chunk17单列；主指标使用原始GT。新输出独立，不覆盖9节诊断。W&B必须online，并以API核实真实step/loss后记录链接。实际启动记录如下。


实际启动：2026-10-01 10:46（北京时间），代码commit `d8a3ee9dc443fc90f8b7df9297f048bbd559b24a`，已push到origin/ar-video-action。四节点启动前8卡均空闲且无计算进程，NCCL走eth0/TCP、IB关闭。两组实际optimizer分组校验通过：基础初始LR≈2e-5，五个指定组≈1e-4；warmup起点实际LR=0，符合原配方。

| 运行 | 输出根目录 | 在线W&B |
|---|---|---|
| 16窗口过拟合 | `outputs/joint_video_hand_pose/ar_v0_2/overfit16_20261001T1045` | [overfit16_clean_history](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/uksp0fsg) |
| 历史视频加噪 | `outputs/joint_video_hand_pose/ar_v0_2/history_video_noise_20261001T1045` | [history_video_noise_p50_s020](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/rek6cap5) |

W&B API已返回两组真实iteration及`loss/video_raw`、`loss/action_raw`、`loss/total`和8项字段loss，证据在准备目录`wandb_verification.json`；不只是本地run ID。10:57检查：过拟合17/300，video/action=0.2233/0.1719，global batch16，约21.1s/同步训练步，峰值allocated49.27/reserved58.56GiB；加噪15/1000，video/action=0.2683/0.2651，global batch39（动态packing），约24.8s/同步训练步，峰值allocated55.72/reserved71.64GiB。两组均无stop_reason；warmup裁剪率100%按原规则仅记录，不能把前期loss变化当作效果验收。

评测同时在Tdebug4/Tdebug5空闲8卡启动原step1000的σ=.05/.1网格；剩余σ=.2、训练窗口对照及新checkpoint评测由同目录`evaluation_plan.json`冻结，`eval_watcher.py`核实官方保存完成标记与本节点GPU空闲后派发原生项目评测入口。输出在`eval/<group>/`，完成标记、逐窗口指标和视频均独立保存；当前尚无完整评测结论。每100步保存的过拟合checkpoint会分别验收，最终两组四个σ和三种历史模式均需完成，不能用单组/单窗口代替全网格。线程继续每30分钟跟进并同步本节；后台检测逐步执行，不受汇报频率影响。

### 10.1 11:36进度与原模型历史噪声网格（2026-10-01）

两组正常运行，未发现STOPPED或节点失败退出。下表为北京时间11:36的完整步；速度取最近10步同步训练段均值，ETA不含后续保存/评测。两组最近10步裁剪触发率均为0%，常驻显存约15.13/15.20GiB，没有持续上涨。

| 组 | 步数 | 秒/步 | 剩余分钟 | video / action / total | global batch | 当前步峰值 allocated / reserved GiB |
|---|---:|---:|---:|---|---:|---:|
| overfit16 | 122/300 | 21.06 | 62 | 0.0984 / 0.0562 / 0.1546 | 16 | 49.27 / 58.56 |
| history_video_noise | 104/1000 | 24.99 | 373 | 0.1318 / 0.0769 / 0.2087 | 33 | 64.36 / 75.44 |

过拟合每rank每步1个clip，全局16；加噪最新每rank[3,1,3,1,2,2,2,1,2,1,2,2,3,1,4,3]，最近10步全局26…47。过拟合iter_000000100已由官方保存并更新latest指针，Tdebug5启动固定16训练窗口gt/oracle评测（11:36完成8窗口，其余运行）；加噪组尚未到首个500步保存点。W&B API再次确认两组running且真实video/action/total和8项字段loss均上传，抽样读取不等于每步即时上传证明。

| action字段loss | overfit step122 | history-noise step104 |
|---|---:|---:|
| 相机平移 | 0.007589 | 0.009834 |
| 相机旋转 | 0.003387 | 0.002154 |
| 右腕平移 | 0.049955 | 0.069912 |
| 右腕旋转 | 0.045256 | 0.063583 |
| 右手形 | 0.083710 | 0.110363 |
| 左腕平移 | 0.041112 | 0.066963 |
| 左腕旋转 | 0.037651 | 0.071420 |
| 左手形 | 0.075483 | 0.097582 |

**以下仅是旧正式模型step1000的推理端噪声扫描，不是新加噪训练的结果。** 同8窗口、seed42、C4联合30步/shift5/CFG1，全部12组完成。σ=0复用冻结基线，σ>0的9组首chunk（没有历史可加噪）与各自基线的相机、双腕、手形和像素MSE最大差异均为0。每组136块，均32个action和16个future视频采样帧；第17块8个样本。按样本数重新汇总并核对summary，PSNR从平均MSE换算。位置和手形为mm，边界为像素MAE。

| Mode | sigma | Camera mm | R/L wrist mm | R/L shape mm | PSNR dB | Boundary MAE |
|---|---:|---:|---:|---:|---:|---:|
| gt | 0 | 31.15 | 57.33/65.03 | 8.42/8.36 | 16.93 | 25.10 |
| gt | 0.05 | 59.08 | 95.44/106.03 | 11.57/11.20 | 14.28 | 40.71 |
| gt | 0.1 | 55.60 | 92.60/100.90 | 11.30/11.11 | 14.20 | 41.07 |
| gt | 0.2 | 58.17 | 94.50/102.63 | 11.36/10.96 | 14.06 | 41.86 |
| pred_history | 0 | 95.23 | 116.36/132.35 | 13.29/11.50 | 12.34 | 48.10 |
| pred_history | 0.05 | 158.69 | 169.75/161.05 | 12.63/10.88 | 12.71 | 50.03 |
| pred_history | 0.1 | 108.22 | 140.19/131.27 | 12.90/11.16 | 12.44 | 51.18 |
| pred_history | 0.2 | 116.97 | 153.95/145.69 | 13.72/12.20 | 12.02 | 53.30 |
| generated | 0 | 219.97 | 243.13/281.79 | 18.93/18.08 | 12.22 | 1.91 |
| generated | 0.05 | 554.72 | 600.28/564.46 | 19.77/19.27 | 12.12 | 1.37 |
| generated | 0.1 | 586.46 | 603.61/600.52 | 20.46/19.44 | 11.84 | 1.09 |
| generated | 0.2 | 748.68 | 796.06/745.36 | 21.73/19.54 | 11.45 | 1.04 |

不动基线（每块GT起点重置）相机/右腕/左腕25.77/44.71/59.67mm。generated位置指标已含累积漂移，不动基线不是同口径的无重置自由rollout基线。

| 第17块模式 | σ | 相机mm | 右/左腕mm | 右/左手形mm | PSNR | 边界MAE |
|---|---:|---:|---:|---:|---:|---:|
| gt | 0 | 23.50 | 43.95/37.34 | 6.39/7.91 | 17.43 | 25.43 |
| gt | 0.05 | 24.53 | 59.42/71.25 | 8.96/9.54 | 13.04 | 47.18 |
| gt | 0.1 | 22.63 | 55.43/64.13 | 8.46/9.54 | 13.15 | 46.11 |
| gt | 0.2 | 20.62 | 56.51/64.98 | 8.99/10.48 | 13.41 | 44.74 |
| pred_history | 0 | 44.80 | 85.94/94.52 | 15.60/11.97 | 12.23 | 52.31 |
| pred_history | 0.05 | 125.12 | 115.13/132.81 | 10.93/9.79 | 12.97 | 52.11 |
| pred_history | 0.1 | 52.33 | 100.65/100.97 | 13.27/9.67 | 12.58 | 53.38 |
| pred_history | 0.2 | 53.12 | 99.11/97.90 | 13.95/10.41 | 12.02 | 56.74 |
| generated | 0 | 327.86 | 341.19/436.31 | 21.87/22.39 | 12.06 | 2.15 |
| generated | 0.05 | 901.34 | 871.53/902.87 | 18.74/20.31 | 12.07 | 1.58 |
| generated | 0.1 | 857.17 | 817.62/881.65 | 18.97/21.15 | 11.75 | 0.95 |
| generated | 0.2 | 1130.26 | 1119.93/1108.15 | 24.61/20.89 | 11.28 | 0.81 |

结论：**旧模型仅在推理端加历史噪声没有改善动作误差，自回归漂移明显加大；不能据此判断新训练的历史增强方案是否有效。** pred_history在σ=.05的视频PSNR略高，但相机和双腕变差。generated边界MAE下降不代表准确，块首图像对GT的MAE从42.46升至47.88/49.59/51.69。不能用边界平滑或训练loss降低代替动作验收。

旧step1000在16训练窗口对照已完成：gt相机/右腕/左腕26.34/52.86/39.23mm，右/左手形8.68/7.62mm，PSNR17.77；oracle为13.06/37.44/31.85mm、8.66/8.00mm、33.35dB。不动基线20.31/45.69/31.25mm。新过拟合第100步评测截至11:36尚未完成，暂不判断通过。

产物在准备目录`baseline_history_noise_report/`：comparison.json、curves.csv、gt/pred_history/generated.png及svg。固定前三窗口视频在`eval/baseline_step1000_sigma{005,010,020}/{gt,pred_history,generated}/window00…02.mp4`，没有挑选。本地镜像`eval_videos/ar_v0.2/training_diagnostics_20261001/baseline_history_noise_report/`含曲线、27段新增视频及index.html；旧σ0视频复用已有本地文件。新模型相同网格完成后再和本批基线同图对照。

### 10.2 过拟合第100步验收（2026-10-01 11:42完成）

固定16训练窗口全部完成gt/oracle，272个完整chunk；原始GT指标，C4联合30步，seed42。按相同窗口和样本数汇总，PSNR由平均像素MSE换算。

| 范围 / 模式 / 模型 | 相机mm | 右/左腕mm | 右/左手形mm | PSNR dB | 边界MAE |
|---|---:|---:|---:|---:|---:|
| all / gt / 原step1000 | 26.34 | 52.86/39.23 | 8.68/7.62 | 17.77 | 22.71 |
| all / gt / 过拟合step100 | 25.03 | 56.32/36.66 | 10.89/9.91 | 18.68 | 19.40 |
| all / oracle / 原step1000 | 13.06 | 37.44/31.85 | 8.66/8.00 | 33.35 | 2.50 |
| all / oracle / 过拟合step100 | 28.83 | 71.13/52.36 | 14.67/13.97 | 33.35 | 2.50 |
| chunk17plus / gt / 原step1000 | 16.35 | 34.24/30.75 | 7.90/6.78 | 18.82 | 19.87 |
| chunk17plus / gt / 过拟合step100 | 16.34 | 33.75/30.89 | 9.63/9.52 | 19.22 | 18.35 |
| chunk17plus / oracle / 原step1000 | 8.40 | 33.64/31.85 | 8.22/7.24 | 33.67 | 2.47 |
| chunk17plus / oracle / 过拟合step100 | 25.34 | 44.19/49.94 | 12.35/13.49 | 33.67 | 2.47 |

不动基线相机/右腕/左腕20.31/45.69/31.25mm。**第100步尚未通过过拟合验收：gt动作三项均不如不动，oracle动作也未学好。** gt视频PSNR较旧模型提高约0.90dB、边界MAE下降，但手形更差。oracle视频是GT经VAE重建，PSNR不代表checkpoint的视频生成能力。保持原计划继续200/300步验收，未调参或重训。

准备目录`overfit100_report/`含comparison.json、gt/oracle.png及svg，272块汇总、第17块16样本与逐块曲线。固定前三训练窗口视频位于`eval/{baseline_step1000_train16,overfit16_step100_train}/{gt,oracle}/window00…02.mp4`。本地镜像`eval_videos/ar_v0.2/training_diagnostics_20261001/overfit100_report/`含曲线、12段视频和index.html。

11:47补充进度：过拟合151/300（50.3%），最近10步21.08s/步，预计余52分钟；加噪129/1000（12.9%），25.44s/步，预计余6小时9分钟。两组最近10步裁剪率均0%，未见stop_reason。最新global batch为16/47；峰值allocated为49.27/55.62GiB，reserved为58.56/75.44GiB；常驻显存约15.13/15.19GiB稳定。video/action/total分别0.094753/0.023229/0.117982和0.136713/0.109695/0.246408。

| 最新action字段loss | 过拟合151 | 加噪129 |
|---|---:|---:|
| 相机平移 | 0.007265 | 0.016831 |
| 相机旋转 | 0.003594 | 0.002616 |
| 右腕平移 | 0.028699 | 0.118356 |
| 右腕旋转 | 0.018100 | 0.099087 |
| 右手形 | 0.034658 | 0.131245 |
| 左腕平移 | 0.024938 | 0.129797 |
| 左腕旋转 | 0.014017 | 0.103385 |
| 左手形 | 0.027148 | 0.150563 |

### 10.3 例行检查（最新：2026-10-01 13:48，北京时间）

过拟合300步已完成，两端退出码0、最终checkpoint及latest指针完整。加噪组正常继续，未见STOPPED或stop_reason；最近10步裁剪率两组均0%，常驻显存稳定。未调参、恢复或重复启动训练。

| 组 | 步数 | 近10步秒/步 | 预计训练剩余 | video / action / total | global batch | 当前/最后步allocated / reserved GiB |
|---|---:|---:|---:|---|---:|---:|
| overfit16 | 300/300 | 22.50 | 0分钟 | 0.065258 / 0.002351 / 0.067610 | 16 | 49.27 / 58.56 |
| history_video_noise | 410/1000 | 25.81 | 254分钟 | 0.137758 / 0.098027 / 0.235785 | 40 | 55.62 / 75.94 |

过拟合显存是最后训练步记录，当前训练GPU已释放；加噪ETA不含保存和评测。加噪最新逐rank clip数：[2, 4, 3, 3, 2, 1, 2, 3, 1, 1, 3, 4, 3, 3, 2, 3]。常驻显存近10步范围：[15.19094467163086, 15.201520919799805]GiB。首次保存设在500步，当前尚未到达。

| action字段loss | 过拟合300 | 加噪410 |
|---|---:|---:|
| 相机平移 | 0.001546 | 0.007955 |
| 相机旋转 | 0.000914 | 0.002195 |
| 右腕平移 | 0.003763 | 0.078062 |
| 右腕旋转 | 0.001853 | 0.065590 |
| 右手形 | 0.002949 | 0.122432 |
| 左腕平移 | 0.003000 | 0.093938 |
| 左腕旋转 | 0.001718 | 0.083392 |
| 左手形 | 0.002531 | 0.153608 |

W&B API已核实加噪running、iteration 403的video/action/total及全部8字段；过拟合finished、iteration300全部字段已在wandb_verification_1259.json核实。一次SDK多字段查询缺少右腕旋转，后续SDK认证检查遇到service process busy超时；改用只读GraphQL复核300步右腕旋转为0.001852999092079699，未发生日志漏记。证据：wandb_verification_direct_followup.json、wandb_verification_noise_latest.json。

[过拟合W&B](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/uksp0fsg) · [加噪W&B](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/rek6cap5)。

Tdebug1/3/4/5上的评测watcher正常，过拟合heldout四档网格均完成，目前等待加噪1000步checkpoint；没有重复派发。训练完成后才可判断加噪训练效果。状态快照：heartbeat_20261001T134848.json。

### 10.4 过拟合200/300步验收（2026-10-01）

两检查点固定16训练窗口gt/oracle全部完成。每模式272块，均32个action和16个future视频采样帧；第17块16样本。重新按样本数汇总，与原summary数值核对通过；PSNR由平均MSE换算。所有位置/手形指标对原始GT关键点计算。

| 范围 / 模式 / 模型 | 相机mm | 右/左腕mm | 右/左手形mm | PSNR | 边界MAE |
|---|---:|---:|---:|---:|---:|
| all / gt / 原step1000 | 26.34 | 52.86/39.23 | 8.68/7.62 | 17.77 | 22.71 |
| all / gt / 过拟合step100 | 25.03 | 56.32/36.66 | 10.89/9.91 | 18.68 | 19.40 |
| all / gt / 过拟合step200 | 12.45 | 23.91/15.62 | 4.66/4.39 | 21.90 | 13.23 |
| all / gt / 过拟合step300 | 7.83 | 14.44/11.48 | 3.71/3.68 | 26.40 | 7.30 |
| all / oracle / 原step1000 | 13.06 | 37.44/31.85 | 8.66/8.00 | 33.35 | 2.50 |
| all / oracle / 过拟合step100 | 28.83 | 71.13/52.36 | 14.67/13.97 | 33.35 | 2.50 |
| all / oracle / 过拟合step200 | 13.87 | 30.46/21.20 | 6.43/6.09 | 33.35 | 2.50 |
| all / oracle / 过拟合step300 | 11.02 | 22.21/17.13 | 5.16/5.02 | 33.35 | 2.50 |
| chunk17plus / gt / 原step1000 | 16.35 | 34.24/30.75 | 7.90/6.78 | 18.82 | 19.87 |
| chunk17plus / gt / 过拟合step100 | 16.34 | 33.75/30.89 | 9.63/9.52 | 19.22 | 18.35 |
| chunk17plus / gt / 过拟合step200 | 9.24 | 22.10/14.31 | 4.27/4.11 | 23.91 | 11.36 |
| chunk17plus / gt / 过拟合step300 | 5.60 | 10.87/8.47 | 3.43/3.48 | 28.33 | 6.24 |
| chunk17plus / oracle / 原step1000 | 8.40 | 33.64/31.85 | 8.22/7.24 | 33.67 | 2.47 |
| chunk17plus / oracle / 过拟合step100 | 25.34 | 44.19/49.94 | 12.35/13.49 | 33.67 | 2.47 |
| chunk17plus / oracle / 过拟合step200 | 14.36 | 22.86/21.71 | 5.77/5.70 | 33.67 | 2.47 |
| chunk17plus / oracle / 过拟合step300 | 9.68 | 15.90/11.96 | 4.70/4.59 | 33.67 | 2.47 |

不动基线all：相机/右腕/左腕20.31/45.69/31.25mm；第17块14.26/33.34/33.12mm。200步gt三项位置均优于不动，300步进一步改善，手形降至右/左3.71/3.68mm，视频PSNR26.40dB，边界MAE7.30。第17块同样改善，没有观察到这16训练窗口在首次历史淘汰处的误差跃升。

结论：**模型能明显拟合这16个训练窗口，当前证据不支持完全学不动；但训练集改善不能替代heldout泛化、生成历史鲁棒性或自由rollout验收。** 第100步未学好的现象到200/300步有实质改善，不能据早期loss或单checkpoint下最终结论。oracle动作到300步也改善，但仍比本批gt联合推理误差高，不据此单独归因。oracle的33.35dB来自GT视频VAE重建，不能当作该checkpoint的生成PSNR。

产物：准备目录overfit200_report/和overfit300_report/的comparison.json、gt/oracle.png/svg。300报告曲线含原step1000、过拟合100/200/300及不动基线；固定前三训练窗口各模式视频完整保留，未挑选样本。本地eval_videos/ar_v0.2/training_diagnostics_20261001/overfit300_report/index.html汇总全部表格、曲线与24段对比视频（复用原基线/100步文件）；新增200/300步共12段视频。heldout四档噪声仍在并行评测，结果独立报告。

### 10.5 过拟合300步：固定8个heldout窗口噪声网格（2026-10-01）

**范围纠正（用户2026-10-01最新指令）：过拟合模型的heldout评测未获用户要求，是执行时错误扩展的范围。下文仅保留已经产生的真实结果，不能视作授权继续此项。定时跟进已暂停，Tdebug1/3/4/5的自动评测watcher均已停止；加噪训练继续，训练进程内异常保护及在线日志保留。之后等用户主动询问，不再自动检查、汇报或派发后续评测。**

这里是16训练窗口过拟合模型，**不是尚在训练的history_video_noise模型**。12组（σ=0/.05/.1/.2 × gt/pred_history/generated）全部完成，并与原step1000的12组逐块同图对照。冻结8窗口hash为8ac5c99d0a80f963b03858dd9260e12aa3cccead0e0bf4475427bd863ec1f03d，seed42、C4联合30步、shift5/CFG1不变；主指标为原始GT关键点。旧模型与过拟合模型的数据量和训练步数不同，此比较不能用于单因素归因。

每组136块、每块32行action和16个future视频采样帧，第17块8个样本。按样本数加权并核对原summary（误差≤1e-9），PSNR从平均像素MSE换算；边界均值不含首块。两模型共18组σ>0对各自σ0的首块主要指标最大差异均0，符合首块无历史可加噪。

| Model | Mode | sigma | Camera mm | R/L wrist mm | R/L shape mm | PSNR | Boundary MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | gt | 0 | 31.15 | 57.33/65.03 | 8.42/8.36 | 16.93 | 25.10 |
| baseline | gt | 0.05 | 59.08 | 95.44/106.03 | 11.57/11.20 | 14.28 | 40.71 |
| baseline | gt | 0.1 | 55.60 | 92.60/100.90 | 11.30/11.11 | 14.20 | 41.07 |
| baseline | gt | 0.2 | 58.17 | 94.50/102.63 | 11.36/10.96 | 14.06 | 41.86 |
| baseline | pred_history | 0 | 95.23 | 116.36/132.35 | 13.29/11.50 | 12.34 | 48.10 |
| baseline | pred_history | 0.05 | 158.69 | 169.75/161.05 | 12.63/10.88 | 12.71 | 50.03 |
| baseline | pred_history | 0.1 | 108.22 | 140.19/131.27 | 12.90/11.16 | 12.44 | 51.18 |
| baseline | pred_history | 0.2 | 116.97 | 153.95/145.69 | 13.72/12.20 | 12.02 | 53.30 |
| baseline | generated | 0 | 219.97 | 243.13/281.79 | 18.93/18.08 | 12.22 | 1.91 |
| baseline | generated | 0.05 | 554.72 | 600.28/564.46 | 19.77/19.27 | 12.12 | 1.37 |
| baseline | generated | 0.1 | 586.46 | 603.61/600.52 | 20.46/19.44 | 11.84 | 1.09 |
| baseline | generated | 0.2 | 748.68 | 796.06/745.36 | 21.73/19.54 | 11.45 | 1.04 |
| overfit300 | gt | 0 | 30.40 | 58.38/68.88 | 8.46/8.08 | 17.50 | 22.92 |
| overfit300 | gt | 0.05 | 34.28 | 64.15/69.49 | 9.27/8.65 | 17.00 | 28.42 |
| overfit300 | gt | 0.1 | 33.48 | 62.69/68.77 | 9.09/8.46 | 16.92 | 28.40 |
| overfit300 | gt | 0.2 | 33.27 | 62.82/68.78 | 8.96/8.36 | 16.73 | 28.44 |
| overfit300 | pred_history | 0 | 31.11 | 69.14/64.74 | 9.83/9.37 | 13.51 | 37.78 |
| overfit300 | pred_history | 0.05 | 40.29 | 65.33/69.15 | 10.77/9.89 | 13.70 | 38.58 |
| overfit300 | pred_history | 0.1 | 41.17 | 66.05/68.64 | 10.65/10.05 | 13.80 | 37.71 |
| overfit300 | pred_history | 0.2 | 40.11 | 68.98/68.83 | 10.93/10.35 | 13.74 | 37.58 |
| overfit300 | generated | 0 | 116.06 | 390.26/258.01 | 67.46/66.53 | 12.69 | 2.06 |
| overfit300 | generated | 0.05 | 166.53 | 311.86/255.11 | 83.20/63.03 | 13.00 | 1.59 |
| overfit300 | generated | 0.1 | 171.30 | 325.30/271.46 | 88.80/66.63 | 13.09 | 1.39 |
| overfit300 | generated | 0.2 | 168.39 | 331.33/298.45 | 90.95/72.13 | 12.99 | 1.48 |

不动基线（每块GT起点重置）相机/右腕/左腕25.77/44.71/59.67mm；generated含累积漂移，不能把这条基线当作无重置自由rollout的同口径基线。

第16块之后单列：本批只有第17块，以下σ=0，其余噪声档完整数据保存在comparison.json。

| 模型 / 模式（σ=0） | 相机mm | 右/左腕mm | 右/左手形mm | PSNR | 边界MAE |
|---|---:|---:|---:|---:|---:|
| baseline / gt | 23.50 | 43.95/37.34 | 6.39/7.91 | 17.43 | 25.43 |
| baseline / pred_history | 44.80 | 85.94/94.52 | 15.60/11.97 | 12.23 | 52.31 |
| baseline / generated | 327.86 | 341.19/436.31 | 21.87/22.39 | 12.06 | 2.15 |
| overfit300 / gt | 28.69 | 41.42/46.89 | 7.56/8.78 | 17.21 | 24.86 |
| overfit300 / pred_history | 28.47 | 56.70/41.32 | 11.27/11.53 | 13.04 | 41.42 |
| overfit300 / generated | 194.97 | 727.49/430.47 | 124.06/139.44 | 12.16 | 2.08 |

结论：**训练集拟合改善没有转化为整体heldout泛化通过。** σ0的gt双腕58.38/68.88mm，未优于旧模型57.33/65.03mm，也未优于不动；pred_history的相机、双腕、手形和视频指标较旧模型改善，但仍不足以认定自由rollout可靠。generated的相机和左腕改善，右腕反而从243.13升至390.26mm；右/左手形从18.93/18.08升至67.46/66.53mm，第17块达124.06/139.44mm。推理加噪没有稳定修复这一问题，不能只看PSNR或边界更平滑判断成功。保持已授权加噪训练配方，等其独立验收后再决定下一步。

产物：准备目录overfit300_heldout_report/的comparison.json、curves.csv、table.md、三模式png/svg；每张曲线包含原模型和过拟合模型四档σ并标出第17块。固定前三窗口每组视频均保留，新增36段，原模型视频复用。已同步本地eval_videos/ar_v0.2/training_diagnostics_20261001/overfit300_heldout_report/；index.html提供表格、曲线与72个新旧视频入口，无挑样本。

本机查看：[验证集对照网页](http://127.0.0.1:54712/training_diagnostics_20261001/overfit300_heldout_report/)；[16训练窗口300步视频](http://127.0.0.1:54712/training_diagnostics_20261001/overfit300_report/)。localhost链接依赖本机预览服务。

## 11. 视频学习率对齐官方 Nano（2026-10-01，用户新增授权）

依据仓内`packages/cosmos3/examples/toml/sft_config/vision_sft_nano.toml`：全量微调lr=1e-4，注释说明扫参后比2e-5收敛快。本对照仅采用该学习率，不迁移官方其他训练设置。

独立配方`cosmos3_joint_video_hand_pose/configs/ar_v0_2_video_lr1e4.toml`复用原fixed-camera experiment，base lr=1e-4；action2llm、llm2action、action_modality_embed、action_state_embed、vision_condition_embed倍率均1，因此五组绝对峰值lr仍为1e-4。相较原正式组，只有默认主干/视频侧lr提高5倍。warmup100/cosine1000/f_min=.1、1000步、500/1000保存、60K token、273/257/129/65/33档、PCA15与两套统计、原官方Nano初始化、HSDP 8×2、无历史加噪保持一致；不从过拟合或旧正式checkpoint续训。

节点Tdebug1(10.3.12.57 MASTER)+Tdebug3(10.3.12.49)，各8×H800。预检16卡均空闲，CPU配额各120核。官方launch_ar_v0_2.sh入口，NCCL eth0/TCP，IB关闭；其余NCCL调优不新增。输出独立目录`outputs/joint_video_hand_pose/ar_v0_2/video_lr1e4_20261001T140218/`，准备/回执目录`outputs/maintenance/video_lr1e4_20261001T140218/`。启动commit与W&B实际run在启动核实后补录。

启动前lr_preflight.json核对六类峰值lr均1e-4；实际官方优化器在首步之前输出optimizer_lr_groups.json并断言一致。warmup下初始实际lr为0，随后逐步升至峰值，不能把0误判为峰值未生效。CPU回归30 passed/1 skipped；跳过项为现有环境条件项，详见cpu.log。与原正式config.yaml核对：除lr/倍率/运行名称，仅新增已关闭的历史噪声参数和为空的fixed_windows_manifest；配置对象序列化差异单独规范化，不当成训练变更。

前100步重点记录逐步裁剪率和video/action/total及8字段loss；warmup内裁剪只记录，最早110步按最近10步9次规则停机。OOM、NaN、loss发散、常驻显存连续上涨保持原停止规则，异常停本组并报告，不调参重启。原暂停的过拟合评测不恢复。

最终评测仅本对照1000步checkpoint，同原冻结8个heldout窗口（hash 8ac5c99d0a80f963b03858dd9260e12aa3cccead0e0bf4475427bd863ec1f03d），gt/oracle/pred_history/generated、seed42、C4联合30步、shift5、CFG1、推理历史噪声σ=0。和原正式1000步、加噪1000步四模式放同一表，缺失模式须补齐同8窗口；已匹配hash的存档优先复用。主指标原始GT关键点，逐块及第17块单列，固定前三窗口视频。oracle视频是GT经VAE重建，不能当作视频生成成绩。

新增视频诊断口径在出结果前固定：使用存档原始RGB、有效640×360区域、真实源帧时间，不使用带投影的展示视频。全局时间偏移搜索±16个视频采样间隔（源stride=2、30Hz时为±1.067秒），有效图像以area方式缩至160×90加速时间搜索，所有候选使用同一内部帧集合（排除首尾各16个future采样帧），以RGB均方误差最小为最佳；正偏移表示预测滞后GT。报告零偏移与最佳MSE、搜索边界命中/平坦曲线及秒数；这只是光度匹配诊断，不能单独证明速度或因果关系。

光流使用同一OpenCV Farneback配置在640×360有效区域计算预测与GT相邻future采样帧，仅统计块内相邻帧，排除块首条件和跨块重置。报告GT运动像素（幅值>0.1像素/采样间隔）上的预测/GT幅值总和之比、双方有效流向（预测幅值>1e-4）的平均余弦及覆盖率/样本数；静止或预测零流不伪造方向值。全体/逐块/第17块分别聚合，幅值比以总和之比、余弦以有效像素数加权；不会对每窗口比值简单平均。全图光流同时包含相机和场景运动，不能直接等同于手速。不同checkpoint/history禁止混合汇总，重复窗口报错。

启动已核实：commit `a32abb74b7d81e4ce7347ce7dcd3af840a0d2478`已push；两节点复查空闲后于14:11起启动官方入口。[本次W&B：mnx2ginp](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/mnx2ginp)，通过API读到第8步11项真实loss。实际优化器因全部lr相同合并为一个412参数组，含default及全部五个显式tag，initial_lr=FP32(1e-4)，启动actual_lr=0符合warmup；断言通过。

14:35检查到42/1000步，近10步25.17秒/步，裁剪率0%；video/action/total=0.136708/0.112828/0.249536，global batch46，当前峰值allocated55.62/reserved74.77GiB，常驻15.19GiB；未见STOPPED。全程截至该步最高allocated见历史日志，不能把当前步值称为全程最大。8字段：相机平移/旋转0.024312/0.004883，右腕平移/旋转0.128137/0.098128，右手形0.126111，左腕平移/旋转0.134332/0.117389，左手形0.157121。早期loss下降不是最终效果验收。

新增评测实现`cosmos3_joint_video_hand_pose/src/ar_v02_video_diagnostics.py`仅离线读取官方存档，不改训练或采样。8项CPU回归通过；两条不同GT窗口真实NPZ以2进程完成短测，用时45.37秒，每窗口255对块内光流、第17块15对，时间搜索/并行聚合可运行。早期混合模式试算仅用于开发检查，不作结果表；正式汇总已加同checkpoint/history及窗口去重断言。代码与测试另行提交，训练仍记录原启动commit。

准备目录evaluation_plan.json冻结原正式组四模式32个现有存档；按sample_id、source_offset、seed、C4/30步逐个核对，并记录metadata hash。新eval_group.py复用原runner，只调整独立输出根目录和显式--toml；仍走官方ar_v02_eval sample/evaluate/overlay。待本次1000步完成后才派发本组和加噪组四模式并做三组统一表；新增视频诊断每种模式用--workers并行，不改变推理配置。过拟合评测不会恢复。

本次异常跟进仅针对视频lr新组：warmup期间每5分钟后台查看，进程内异常仍逐步检测；健康时不逐轮通知，第100步汇报一次，之后每30分钟检查，异常/完成及时报告。旧组的定时汇报及旧eval_watcher保持关闭；本次自动跟进仅按本节新授权执行。
