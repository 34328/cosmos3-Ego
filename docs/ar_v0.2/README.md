# AR v0.2（待开始）

v0.1 的结论和改进方向见 [../ar_v0.1/experiment.md](../ar_v0.1/experiment.md) 第 6 节。初始化统一用官方原始 Cosmos3-Nano：`/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp`。

## 待解决

### 训练多样本 packing

**现状**：v0.1 每步每卡只放 1 个 clip（`max_samples_per_batch=1`），没用上 Cosmos `PackingDataLoader` 的动态 packing。

| clip | token 数（干净 + 带噪） | 占 75K 上限 |
|---|---|---|
| T=129 | 约 16.4K | 22% |
| T=65 | 约 8.4K | 11% |
| T=33 | 约 4.4K | 6% |

有效 batch 只有 8 个 clip / 步。每卡显存已用约 55 GiB（大部分是权重、优化器状态和激活），能多放几个 clip 要实测。

**原因**：
1. 官方 `OmniMoTCausalModel` 的 replayed teacher forcing（先干净前向缓存 K/V，再带噪前向）只支持一个 batch 一个样本，多样本会直接报错。
2. `src/ar_attention.py` 的 lingbot mask 按单个 clip 生成 block id，没有样本编号，多样本拼接后会互相看到。

**要改**：
1. 注意力 mask 加一项"同一样本才能互相看到"（FlexAttention 支持，改动小）。
2. 官方 replay 路径改成按样本分段处理干净/带噪两次前向和缓存的 K/V（框架补丁，工作量较大）。每步随机的 C 和窗口可以全 batch 共用，也可以按样本分别设。

**过渡办法**：梯度累积可以扩大有效 batch，但不提高单卡利用率。
