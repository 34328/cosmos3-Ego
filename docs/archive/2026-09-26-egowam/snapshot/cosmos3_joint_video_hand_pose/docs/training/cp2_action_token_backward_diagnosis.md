# Joint 训练踩坑与工程 Guardrail

> 这里只保留已经被对照实验坐实、后续仍需遵守的结论。失败 checkpoint、
> smoke、fixed-pack 临时输出和原始诊断日志均已删除；需要复查时从 base
> checkpoint 重新运行最小实验。

## 必须遵守

| 项目 | 当前约束 | 原因 |
|---|---|---|
| joint 并行 | CP=1, FSDP=8 | has_action=True + CP=2 会令中前层参数梯度 non-finite，清洗后表现为不更新 |
| token cap | 75000 | 8×H800 已验证；继续上调显存余量不足 |
| allocator | PYTORCH_ALLOC_CONF=expandable_segments:True | 动态 pack 会产生显存碎片 |
| 梯度诊断 | 同时记录 raw NaN/Inf 数量与裁剪次数 | GradClip(force_finite=True) 会先把 non-finite 清零，小 grad norm 不等于健康 |
| mask 配置 | 必须验证配置已经透传到 Cosmos3VFMNetworkConfig | 只在 experiment 层写开关曾导致 mask 实际未启用 |
| 动态 Flex | action-query 使用独立 torch.compile(dynamic=True) wrapper | 静态 wrapper 每种 pack shape 重编译，达到 8 个图后 fallback 并 OOM |
| 时间对齐 | A_t -> V_<=floor(t/4), A_<=t | causal VAE 的 V_i 最晚依赖原始帧 4i；ceil(t/4) 会泄露未来帧 |

## 已坐实的边界

- pure IT2V 在 CP=2 正常；加入 action token 行后 CP=2 异常。把 action loss、
  action value、action query 或 action→video 通路分别关闭仍不能消除，因此根因不是
  LR、困难样本或 action loss 权重，而是 joint token layout 与 CP backward 的交互。
- 修为 CP=1/FSDP8 后，75K 正式基线训练到 1200 step，没有旧式 video-loss
  灾难性崩升。
- modality mask 修复透传后，fixed-pack action 清零/反转对 video 输出的差异为 0，
  action 输出仍变化，证明 video 不再读取 action。
- B3 将 camera/wrist rigid pose 改成 frame-to-frame 增量后，四个 replay 的低频
  手腕漂移明显减弱；后续 joint 实验继续使用 B3。
- temporal mask 的动态 kernel 已通过 10 种 shape 的 forward/backward，以及
  8 卡 FSDP 12-step 真实 pack smoke 和完整 checkpoint 保存。

## 遇到异常时的最小顺序

1. 先检查 raw non-finite fraction 和各层参数是否真实更新。
2. 固定同一个 pack、sigma、noise 做重复/清零/反转 action intervention。
3. 比较 CP1 与 CP2；不要先归因于 LR 或样本难度。
4. 检查 mask 开关的最终 resolved config、实际 network config 和可见性矩阵。
5. 动态 attention 必须跨过 8 种 shape，再检查是否出现 fallback 或 dense OOM。

在 CP2 的底层 DTensor/attention backward 被正式修复前，不再启动 joint CP2
长训练，也不保留其临时产物。
