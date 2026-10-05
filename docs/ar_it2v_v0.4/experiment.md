# V0.4 实验记录

## 实现验收（正式训练尚未启动）

2026-10-05：用户确认按LingBot方式拆分GT历史与预测表示，整段所有预测块并行监督。历史噪声保持已选LingBot公开代码配方，预测块改为Cosmos3官方Nano waver/480shift5；不再使用项目自写uniform/clamp。数据源预算65536，其余已批准训练参数见 [design.md](design.md)。用户拒绝CPU offload，本版未启用。

Tdebug6、cosmos3环境、`CUDA_VISIBLE_DEVICES=''`、仓库根＋packages/cosmos3 PYTHONPATH、CPU线程1：9项针对真实pack/noising/loss入口和历史梯度的临时测试通过（26.21秒）。首次检查发现原生视频pack也有文本CE索引，已修正为扩展双流后的正确索引映射，保留标签/权重与原pack不变；修复后重跑9项全通过。测试和日志仅放ignored `tmp/ar_it2v_v04/`。

独立GPT-6 Astra/xhigh审查最新实现，未发现剩余阻塞性代码问题。核对H/P可见性、绝对RoPE、逐块官方timestep、原生输出/尾权重/分母与样本平均、历史梯度、RNG恢复及推理兼容。CPU测试中的denoise替身网络不能代替真实FlexAttention/full AC/FSDP GPU前反传；显存与性能尚未验收，不构成正式启动回执。
