# AR v0.1：分块因果自回归视频–动作联合生成

状态：**已完成（2026-09-27）**。v0.2 另开目录。

| 文件 | 内容 |
|---|---|
| [design.md](design.md) | 方案（2026-09-26 确认）；第 11 节为实现记录、已知偏差、推迟项 |
| [experiment.md](experiment.md) | 训练、三种条件的回放评测、手部投影、推理耗时、结论与 v0.2 方向 |
| [review_codex_2026-09-27.md](review_codex_2026-09-27.md) | Codex 外部审阅（架构、实现问题、loss 尺度建议） |

**一句话结论**：链路（训练 → 逐 chunk 先视频后动作推理 → 评测 → 投影回放）已跑通。视频生成中规中矩。动作在读取真实视频时学得好，换成生成视频后误差回到"不动"的水平。第 1 个 chunk 最差，逐帧增量从首帧积分会累积漂移。v0.2 重点改动作。

**代码入口**（`cosmos3_joint_video_hand_pose/`）：

| 用途 | 位置 |
|---|---|
| 数据 | `src/ar_dataset.py` |
| 注意力 | `src/ar_attention.py` |
| 模型 | `src/ar_model.py` |
| 推理 / 评测 | `src/ar_inference.py` |
| 手部投影 | `src/ar_overlay.py` |
| 耗时测量 | `src/ar_benchmark.py` |
| 配置 | `configs/ar_v0_1.toml`（默认从官方 Cosmos3-Nano DCP 初始化） |
| 启动 | `scripts/launch_ar_v0_1.sh`、`scripts/launch_ar_v0_1_smoke.sh` |

**产物**：
- 权重：`outputs/joint_video_hand_pose/ar/ar_v0.1_sft48464/checkpoints/iter_000001200`（这次训练的初始化不是官方权重，见 design.md 第 11 节）
- 评测：`outputs/joint_video_hand_pose/ar/eval/`
