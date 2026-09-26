# Cosmos 3 Joint Video + Hand Pose 文档

当前可解释的小数据联合生成基线从 **CP=1 / FSDP=8 / 75K** 开始。更早的
`overfit_v0.0`、CP=2 joint 和两次 video-loss 崩升只作为历史诊断，不再作为可运行基线。

## 当前基线

- [当前联合 Overfit 基线](training/current_joint_overfit_baseline.md)：稳定 CP1 run 的数据、优化器、loss、checkpoint 与 replay。
- [模型/数据学习合同](model_learning/overfit_v0.0_contract.md)：输入、57D action、坐标系、normalizer、loss、visibility、冻结范围和 replay 合同；文件名保留历史版本号。
- [数据流图](model_learning/pipeline.md)：从 EgoVerse segment 到 Cosmos packed batch、action decode 和 replay。
- [CP2 × action token backward 诊断](training/cp2_action_token_backward_diagnosis.md)：解释为何正式 joint run 禁用 CP=2。

## 专项规范

- [Hand visibility loss v0.6](specs/hand_visibility_loss_v0.6.md)
- [视频训练链路 v1](specs/todo4_video_training_pipeline_v1.md)

## 后续实验

- [Video-First WAM modality-causal mask v1](future_experiments/video_first_wam_mask_v1.md)：已实现并正在 CP1 稳定底座上验证。

Artifact 目录中的 `cosmos3_action_contract/v2` 和 `cosmos3_hand_codecs/v2_4` 是不可变数据格式版本，不是实验版本号；为保持既有 checkpoint 可解码，不随 overfit 实验重命名。
