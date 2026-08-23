# Cosmos 3 Joint Video + Hand Pose 文档

当前唯一有效的小数据过拟合基线是 **overfit_v0.0**。历史 V1–V6 已合并，不再作为可运行配置保留。

## 当前基线

- [模型学习合同](model_learning/overfit_v0.0_contract.md)：输入、57D action、坐标系、normalizer、loss、visibility、冻结范围和 replay 合同。
- [数据流图](model_learning/pipeline.md)：从 EgoVerse segment 到 Cosmos packed batch、action decode 和 replay。
- [训练运行合同](training/overfit_v0.0.md)：36 episodes / 181 segments 的最终 overfit 配置、checkpoint 和推理路径。
- [机器可读配置](../configs/overfit_v0_0.yaml)：供审计使用；实际 Cosmos 入口读取同目录 TOML。

## 专项规范

- [Hand visibility loss v0.6](specs/hand_visibility_loss_v0.6.md)
- [视频训练链路 v1](specs/todo4_video_training_pipeline_v1.md)

## 后续实验

- [Two-stage IT2V → Joint](training/two_stage_it2v_then_joint_v1.md)
- [Future experiments](future_experiments/README.md)：尚未进入实现合同的研究想法。

Artifact 目录中的 `cosmos3_action_contract/v2` 和 `cosmos3_hand_codecs/v2_4` 是不可变数据格式版本，不是实验版本号；为保持既有 checkpoint 可解码，不随 overfit 实验重命名。
