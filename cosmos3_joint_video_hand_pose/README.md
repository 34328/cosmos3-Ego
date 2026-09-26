# 视频与双手动作联合训练

已实现 EgoVerse 数据适配、57D action 编解码、Cosmos 联合训练、恢复及预测/GT 四格回放。

- [当前实现与运行说明](../docs/current-state.md)
- [文档导航](docs/README.md)
- [历史实验归档](../docs/archive/2026-09-26-egowam/README.md)

保留的配方为 `configs/overfit_v0_6_frame_delta_temporal_mask.toml`（B3 表示加时间遮罩），入口位于 `scripts/`。v0.0–v0.5 的 TOML 与启动脚本已移除，其 Python 配置仍保留在 `src/config.py` 中作为 v0.6 的继承链；历史内容见 main 分支及归档。新研究使用独立配置与 run 名称。
