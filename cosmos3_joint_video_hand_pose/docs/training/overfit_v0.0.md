# Joint Video + Hand Pose Overfit v0.0

这是最终小任务过拟合基线。旧 V1–V6 配置已经删除；本版本不再通过多层版本继承构造。

## 数据与初始化

```text
base checkpoint   /mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
task subset       brushing_shoes / repair_bench
episodes          36
segments          181
prompt            episode task context + current segment，Cosmos structured JSON
canvas            640x360 -> bottom reflect-pad -> 640x368
temporal          T=1+4n
token cap         85000
```

超限 segment 独立尝试保留原始帧的 80%、70%、60%、50%；首帧与所有逐帧模态使用同一索引。50% 仍超限则丢弃。

## 模型与 loss

- Cosmos WAM：首帧 video 和首帧真实 57D action 为 clean condition，future video/action 为 noisy generation target。
- Reasoner/LLM、Wan VAE、左右 MLP-AE-15 冻结；训练共享 generation pathway 和原生 action projection。
- video loss 完整复用 Cosmos flow matching，系数 `10`。
- action loss 只计算 `[0:57]`，八个 active 子块等权，系数 `7`；`[57:64]` padding 与 slot 0 不计 loss。
- camera 子块始终有效；每只手的 wrist translation、wrist rotation、hand latent 三块共享该侧 GT visibility。`lambda_out_of_fov=0`。
- `sample_level_loss_averaging=True`，跨 rank 得到全局 sample mean。
- action contract 固定为 `artifacts/cosmos3_action_contract/v2`；训练和 replay 启动前校验其哈希。

## 训练配置

```text
world size         8
CP / FSDP shard    2 / 4
precision          BF16
activation ckpt    full
torch.compile      off
EMA                off
base LR            2e-5
action projection  5x = 1e-4
warmup             20 steps
cosine cycle       2000 steps, min ratio 0.1
max iterations     2000
save interval      300
logging            every optimizer step
```

实际配置：

```text
configs/overfit_v0_0.toml
configs/overfit_v0_0.yaml
```

启动：

```bash
cosmos3_joint_video_hand_pose/scripts/launch_overfit_v0_0.sh
```

默认 staged 脚本在 300/600/900/1200/1500/1800/2000 保存完整 DCP，释放训练 ranks 后生成固定四个长 segment 的 replay，再从该 DCP 恢复训练。

## 输出

```text
outputs/joint_video_hand_pose/train/overfit_v0.0/
outputs/joint_video_hand_pose/inference/overfit_v0.0/iter_*/replays/
```

Replay 为 H.264：顶部显示 segment 指令；第一行是生成视频上的预测手投影与预测 F0 三维视图；第二行是 GT 视频上的 GT 手投影与 GT F0 三维视图。

## W&B

每 step 记录 video/action/total loss、两个实际 LR 和八个 action 子块 raw loss。该 overfit 结果只用于验证 pipeline 和可学习性，不作为完整 100h 训练超参数结论。
