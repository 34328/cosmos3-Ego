# Video-first WAM attention 方案

> 当前状态：v0.6 的最小实现及 fixed-pack action intervention 已通过。
> 本方案只改变 action 侧的可见性，不改变原生 IT2V 的 video attention。

## 1. 实验目的

Cosmos joint WAM 在同一 Generator 中预测 future video 与 action。我们要验证的
不是“把所有 token 都改成时间因果”，而是一个更小、更可归因的问题：

- 保留 Cosmos 已有的 IT2V 生成能力；
- 禁止 action token 反向进入 video query；
- 让 action 按时序读取 video 和 action 历史，学习与视频一致的轨迹。

这是生成图上的结构约束，不等价于识别物理世界的因果关系。

## 2. 精确 attention 语义

token 分成两个 GEN 流：

- Video：首帧图像 token I0 + 全部 future-video token；
- Action：clean action a0 + future action A1...AT。

UND/text 继续使用 Cosmos 原有 causal stream。

### Video query

所有 I0 和 future-video query：

- 可读同 sample 的 UND/text；
- 可读 I0 和全部 future-video token，保持原生 IT2V 双向 full attention；
- 不可读任何 action token，包括 a0。

因此首帧并没有被限制为只能看过去；I0 与 future video 仍然彼此可见。

### Action query

对 raw frame t 的 action query A_t：

- 可读同 sample 的 UND/text；
- 可读 video latent V_0...V_floor(t/4)；
- 可读 action A_0...A_t；
- 不可读未来 video latent 或未来 action。

4 是当前 causal VAE 的时间压缩率。使用 floor 而不是 ceil，避免读取包含未来
raw frame 的 video latent。

### 可见性摘要

| Query | 可见 Key |
|:--|:--|
| I0 / future video | text + I0 + 全部 future video |
| A_t | text + V_<=floor(t/4) + A_<=t |

所有规则还必须满足 same-sample；packed sample 之间永远不可见。

## 3. 为什么这样更合理

旧版 C/V/A 三角把 I0 单独视为 clean C，并禁止 I0 query 读取 future video。
这会改变纯 IT2V attention，本身就可能提高 video loss，导致无法判断变化来自
action 隔离还是视频生成器被改坏。

当前两流方案保留 video 内部的完整双向信息，同时彻底删除 video<-action
边。由于任意一层的 video query 都看不到 action，action 也无法通过 I0 这条
中间路径绕回 video。

共享主干参数仍同时接收 video loss 和 action loss 的梯度。这是联合学习的预期
行为：结构约束保证同一次 forward 中 video 不依赖 action 输入，但 optimizer
更新共享参数后，下一 step 的 video 函数仍可能被 action loss 改变。

## 4. 实现边界

- video 分支仍使用 Cosmos varlen FlashAttention；
- 只有 action query 使用时序 FlexAttention；
- action FlexAttention 使用独立的 dynamic-shape compiled wrapper，以适配
  75K 动态 pack，避免每个 shape 重新生成静态图；
- CP 固定为 1，FSDP shard degree 为 8；
- B3 frame-to-frame rigid-pose 表征、LR、loss 权重、normalizer、noise
  schedule、数据和 sampler 均保持不变。

## 5. 已完成验证

fixed-pack intervention 在相同权重、sigma、noise 和 pack 下得到：

- 重复原 action：video relative L2 = 0，max abs = 0；
- future action 清零：video relative L2 = 0，max abs = 0；
- future action 反转：video relative L2 = 0，max abs = 0；
- action loss 对清零/反转分别发生明显变化。

这直接证明 video 输出不读取 action，而 action 流仍然工作。

此外已通过：

- CPU 可见性与跨 sample 隔离测试；
- 10 种动态 action-query shape 的 FlexAttention 测试；
- 8 卡 4-step 真实 forward/backward smoke，梯度 finite、无 NaN/OOM。

## 6. v0.6 判定标准

正式训练需同时检查：

1. video loss 是否回到 B3 / pure-IT2V 附近，而不是旧错误 mask 的高起点；
2. action 各子 loss 是否正常下降；
3. grad norm、clip 次数与 nonfinite 计数；
4. step 1200 的四样本 replay 中，video 质量与 B3 手部轨迹稳定性。

如果 video 质量恢复而 action 对齐改善，才支持继续沿 video-first temporal WAM
方向迭代；否则应分别检查共享梯度冲突与 action temporal alignment，而不是再
同时改 LR 或 loss 权重。
