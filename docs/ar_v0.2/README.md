# AR V0.2

逐块自回归的视频 + 手部动作联合模型。当前方案 `fixed_camera_wrist_local_delta_latent_v1`：57D action，C=4、K=8，15 个历史 chunk，联合去噪 30 步，`L_video + L_action`。

| 文档 | 内容 |
|---|---|
| [design.md](design.md) | 方案定义、公式、验收要求、代码导航（第 7 节） |
| [experiment.md](experiment.md) | 当前结果、评测、负结果、已知问题 |
| [archive/](archive/) | 过程日志与历史 review，只读，不再更新 |

**现状（2026-09-30）**：1000 步双节点正式训练已完成（W&B [bytyfmqa](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/bytyfmqa)）。给真实视频时动作明显优于不动基线；用模型自己生成的视频时不如不动基线。当前瓶颈是视频生成，详见 experiment 第 5、7 节。

配置：`cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml`；启动：`cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_2.sh`。`ar_v0_2.toml` 仅用于旧表示回归。

归档内容：

| 文件 | 内容 |
|---|---|
| [experiment_log_2026-09-28_to_09-30.md](archive/experiment_log_2026-09-28_to_09-30.md) | 原 experiment.md 全文：训练前短测、逐步验收、半小时跟进、旧表示训练、官方复用审计 |
| [review_codex_2026-09-29.md](archive/review_codex_2026-09-29.md) | 动作表示、配方、双节点的逐轮 review 与用户决策 |
| [review_codex_2026-09-28.md](archive/review_codex_2026-09-28.md) | KV cache 与 attention 修复复审 |
| [code_map.md](archive/code_map.md) | 旧代码导航，已并入 design 第 7 节 |

本地 `data_audit/` 保存数据审计资料。
