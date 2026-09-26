# EgoWAM 历史实验归档

归档日期：2026-09-26；清理前 Git HEAD：`599577d`。本归档记录已有工作，不定义 RBS WAM 后续研究路线。

## 实验沿革与结论边界

| 阶段 | 已有结果与限制 |
|---|---|
| 纯 IT2V | 保留 run `pure_it2v_v2_recheck_20260824` 的 300/600 步检查点及回放；作为视频对照。 |
| v0.2 CP1 75K | 1200 步已验收基线；历史文档记录消除了 CP2 的灾难性 video-loss 崩升。 |
| v0.3 | active loss normalization 与独立 action noise schedule 的配方；不虚构独立完成 run。 |
| v0.4 | 历史训练时 mask 配置未透传；其检查点不是有效 mask 对照。当前源码已修复，重跑同名配方也不是复现旧错误。 |
| v0.5 B3 | camera/wrist 改为逐帧 SE(3) 增量；文档记录四训练样本回放中的低频手腕漂移减弱，本次未重看视频。 |
| v0.6 | B3 加 video-first temporal mask；存在 1200 步检查点和四样本回放，优于 B3 尚无充分结论。 |
| 无首帧动作 T-1 | 设计稿，未实现；不得用于旧 checkpoint 的解码或续训。 |

训练数据仅为 36 episodes / 181 train segments；固定四样本也是 train。没有在本次清理中进行新的训练、完整曲线分析或泛化评估。

## 保存的最后一步数值

下表来自本地 W&B summary；不是验证集均值，不代表全程梯度健康。B3 改变了动作表示及 normalizer，action loss 不能直接跨合同比较。

| Run | Step | Video raw | Action raw |
|---|---:|---:|---:|
| overfit_v0.2_lr_balanced_cp1_75k_repro | 1200 | 0.08715 | 0.00985 |
| overfit_v0.4_video_first_causal_mask | 1200 | 0.08728 | 0.00972 |
| overfit_v0.5_frame_delta_b3 | 1200 | 0.09161 | 0.02838 |
| overfit_v0.6_frame_delta_temporal_mask | 1200 | 0.09367 | 0.02777 |

机器可读来源见 [results.json](results.json)。检查点保留于仓库 `outputs/joint_video_hand_pose/overfit/<run>/checkpoints/`；回放保留于 `outputs/joint_video_hand_pose/inference/<run>/iter_000001200/replays/`。本次未重新加载完整 DCP 验证其可恢复性。

## 应继承的工程经验

- joint action + CP2 曾出现非有限梯度；使用 CP1/FSDP8、75K cap。
- 记录原始 NaN/Inf 与裁剪，梯度清洗后的小范数不能证明健康。
- 核实最终网络 mask 配置；动态 FlexAttention 应覆盖多种 pack shape。
- B3 future normalizer 来自 36-episode 子集，扩展数据需重新统计。
- checkpoint 与 codec/normalizer 的自动绑定尚待完善。

## 原始资料

- [联合实验原 README](snapshot/cosmos3_joint_video_hand_pose/README.md)
- [原文档导航](snapshot/cosmos3_joint_video_hand_pose/docs/README.md)
- [原模型与数据合同](snapshot/cosmos3_joint_video_hand_pose/docs/model_learning/overfit_v0.0_contract.md)
- [CP1 基线原记录](snapshot/cosmos3_joint_video_hand_pose/docs/training/current_joint_overfit_baseline.md)
- [CP2 诊断](snapshot/cosmos3_joint_video_hand_pose/docs/training/cp2_action_token_backward_diagnosis.md)
- [时间遮罩方案](snapshot/cosmos3_joint_video_hand_pose/docs/future_experiments/video_first_wam_mask_v1.md)
- [T-1 设计稿](snapshot/cosmos3_joint_video_hand_pose/docs/future_experiments/action_without_initial_state_v1.md)

`snapshot/` 是清理前文档、描述 YAML 和一次性实验链的原始快照；保留旧路径及互相矛盾的历史陈述用于溯源，不能直接作为当前启动说明。一次性 chain 脚本的当前入口已停用。

## 本次清理

- shell/Python 运行入口改为动态解析仓库位置。
- 纯视频回放改用保留的固定输入，支持显式覆盖选择目录。
- 262 个运行 JSON 仅迁移项目根路径；原始备份和逐文件摘要位于 `outputs/maintenance/2026-09-26-path-relocation/`，该目录不随 Git 分发。
- 检查点、媒体、日志、冻结 artifact 均保留原样；不创建兼容性旧根目录软链接。
- 旧详细合同改为归档入口；修正文档导航和资产版本说明。

后续从[当前实现](../../current-state.md)开始。

本次检查结果见[验证记录](validation.md)。
