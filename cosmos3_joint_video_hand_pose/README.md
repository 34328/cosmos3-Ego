# 视频与双手动作联合训练

已实现 EgoVerse 数据适配、57D action 编解码、Cosmos 联合训练、恢复及预测/GT 四格回放。

- [当前实现与运行说明](../docs/current-state.md)
- [文档导航](docs/README.md)
- [历史实验归档](../docs/archive/2026-09-26-egowam/README.md)

配方：`configs/overfit_v0_3_active_norm_independent_action.toml` 为 full-attention 配方，`overfit_v0_5_frame_delta_b3.toml` 为 B3 表示，`overfit_v0_6_frame_delta_temporal_mask.toml` 为 B3 加时间遮罩。对应入口位于 `scripts/`。

v0.0 仅作基础 smoke/audit 配置；v0.4 历史 run 不能证明 mask 效果。一次性 v0.4→v0.5 自动实验链已停用。已有实验和输出保留，新研究使用独立配置与 run 名称。
