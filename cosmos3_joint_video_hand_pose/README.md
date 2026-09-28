# 视频与双手动作联合训练

已实现 EgoVerse 数据适配、57D action 编解码、Cosmos 联合训练、恢复及预测/GT 四格回放。

- [当前实现与运行说明](../docs/current-state.md)
- [文档导航](docs/README.md)
- [历史实验归档](../docs/archive/2026-09-26-egowam/README.md)
- [AR V0.2 逐块图像／state 条件方案与验收状态](../docs/ar_v0.2/README.md)：新入口配置为 `configs/ar_v0_2.toml`，布局 `joint_chunk_cond_v1`，使用 `data_v2` 的块首相机系统计。每块 U/S 条件与 V/A 联合去噪，采样固定 30 步；正式训练前仍须完成文档验收项。旧 `ar_inference` 属于 V0.1，不能用其布局解包新版输出。

非 AR 基线配方为 `configs/overfit_v0_5_frame_delta_b3.toml`（B3 表示，无额外 attention mask），Python 配置见 `src/config.py`（v0.0 → v0.2 → v0.3 → v0.5）。v0.4/v0.6 的 joint video-action mask 实验已随 cf5d68c 同步删除，原实现见 commit `8525625`。AR 视频–动作联合生成见 [`docs/ar_v0.1/`](../docs/ar_v0.1/README.md)：配方 `configs/ar_v0_1.toml`（默认从官方 Cosmos3-Nano DCP 初始化），训练 `scripts/launch_ar_v0_1.sh <toml>`，冒烟 `scripts/launch_ar_v0_1_smoke.sh`，推理与一致性检查 `python -m cosmos3_joint_video_hand_pose.src.ar_inference`。新研究使用独立配置与 run 名称。
