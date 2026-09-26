# Cosmos 3 Joint Video + Hand Pose

当前 joint 小数据训练统一使用 **CP=1 / FSDP=8 / 75K**。

- 文档入口：[docs/README.md](docs/README.md)
- 当前 full-attention 配置：`configs/overfit_v0_3_active_norm_independent_action.toml`
- 当前 full-attention 训练入口：`scripts/launch_overfit_v0_3_active_norm_independent_action.sh`
- 已验收 CP1 checkpoint replay：`scripts/run_current_joint_baseline_replays.sh`
- Video-first causal-mask 配置：`configs/overfit_v0_4_video_first_causal_mask.toml`
- Video-first causal-mask 训练入口：`scripts/launch_overfit_v0_4_video_first_causal_mask.sh`

`overfit_v0_0` 只保留为模型/数据合同和 smoke/audit 的基础配置，不再提供正式训练入口。
历史 CP=2 与 action-loss-off 消融的结论保存在 `docs/training/`，其失败配置不再保留为可运行实验。
