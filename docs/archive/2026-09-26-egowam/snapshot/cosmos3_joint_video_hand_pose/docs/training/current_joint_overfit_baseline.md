# 当前 Joint Video + Hand Pose Overfit 基线

> 状态：2026-08-25 起唯一推荐的 joint 训练底座。
> 根因 guardrail：`has_action=True` 时禁用 CP=2；详见
> [CP2 × action token backward 诊断](cp2_action_token_backward_diagnosis.md)。

## 数据与初始化

```text
base checkpoint   /mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
task subset       brushing_shoes / repair_bench
episodes/segments 36 / 181
prompt            episode context + segment instruction
canvas            640x360 -> bottom reflect-pad -> 640x368
temporal          T=1+4n
token cap         75000
dataloader        in_order=true, stateful=true
```

数据、57D action、visibility、normalizer、MLP-AE-15 和 replay 语义没有因
CP guardrail 改变。`in_order=true` 保证 worker 结果按提交顺序交付并支持状态恢复，
不承诺每个 segment 严格同频率；checkpoint 后 dataloader 从保存状态继续，不会每
300 step 从头开始。

## 模型、Loss 与噪声

- 首帧 RGB `I0` 与真实 action state `a0` 是 clean condition；future video/action
  是 flow-matching target。
- video loss weight `1.0`，action loss weight `0.7`。
- `normalize_loss_by_active=true`，condition、padding `[57:64]` 和画外手无效块不进分母。
- 八个 action 子块等权；`lambda_out_of_fov=0`。
- `independent_action_schedule=true`：video 使用 Waver，action 使用 logitnormal，
  `shift_action=5`。

## 优化与并行

```text
world size         8
CP / FSDP shard    1 / 8
precision          BF16
activation ckpt    full
base LR            2e-5
shared/video       4x = 8e-5
action projection  5x = 1e-4
warmup             100 steps
cosine cycle       1200 steps, min ratio 0.1
GradClip           norm=1, force_finite=true
allocator          PYTORCH_ALLOC_CONF=expandable_segments:True
```

`force_finite=true` 会在 norm/clipping 前把 NaN/Inf 清零，因此必须同时记录 raw
non-finite element count/fraction；清洗后的小 grad norm 不能单独证明梯度健康。

## 已验收 Run

```text
run
  overfit_v0.2_lr_balanced_cp1_75k_repro

checkpoint
  outputs/joint_video_hand_pose/overfit/
    overfit_v0.2_lr_balanced_cp1_75k_repro/checkpoints/iter_000001200

replay
  outputs/joint_video_hand_pose/inference/
    overfit_v0.2_lr_balanced_cp1_75k_repro/iter_000001200/replays
```

W&B: `te75l18r`。step 1→1199 的 video raw 从 `0.2099` 到 `0.1391`，
action raw 从 `0.5053` 到 `0.0151`；video raw 最大单点 `0.2461`，没有复现
CP2 run 中 `0.5–0.7` 的灾难性崩升。step 1200 已保存 checkpoint 并完成固定四个
train segment 的 replay，随后按计划早停。

## 当前实验链

- v0.2 CP1 75K：已验收的 joint 优化/并行基线。
- v0.4：历史 modality-mask 尝试；训练时配置透传尚未修好，不能作为 mask
  有效性的证据，仅保留 replay 作历史视觉对照。
- v0.5 B3：camera/wrist 改为 frame-to-frame rigid delta，四样本 replay 显示
  低频手腕漂移减弱；后续实验沿用 B3。
- v0.6：在 B3 上启用已修复的 temporal mask：
  video 不读 action，A_t -> V_<=floor(t/4), A_<=t。LR、loss、数据、
  1200-step scheduler 与 v0.5 保持不变。
