# V0.4 实验记录

## 实现验收（正式训练尚未启动）

2026-10-05：用户确认按LingBot方式拆分GT历史与预测表示，整段所有预测块并行监督。历史噪声保持已选LingBot公开代码配方，预测块改为Cosmos3官方Nano waver/480shift5；不再使用项目自写uniform/clamp。数据源预算65536，其余已批准训练参数见 [design.md](design.md)。用户拒绝CPU offload，本版未启用。

Tdebug6、cosmos3环境、`CUDA_VISIBLE_DEVICES=''`、仓库根＋packages/cosmos3 PYTHONPATH、CPU线程1：9项针对真实pack/noising/loss入口和历史梯度的临时测试通过（26.21秒）。首次检查发现原生视频pack也有文本CE索引，已修正为扩展双流后的正确索引映射，保留标签/权重与原pack不变；修复后重跑9项全通过。测试和日志仅放ignored `tmp/ar_it2v_v04/`。

独立GPT-6 Astra/xhigh审查最新实现，未发现剩余阻塞性代码问题。核对H/P可见性、绝对RoPE、逐块官方timestep、原生输出/尾权重/分母与样本平均、历史梯度、RNG恢复及推理兼容。CPU测试中的denoise替身网络不能代替真实FlexAttention/full AC/FSDP GPU前反传；显存与性能尚未验收，不构成正式启动回执。

## 首次 65K 双流 GPU 容量短测：OOM，正式训练未启动

2026-10-05 23:58:23 至 2026-10-06 00:03:20（北京时间），源提交 `c352f1b89e9bf5871aa9d315a3f1937abe279afd`。Tdebug3 单节点8×H800，HSDP8×1/CP1，源预算65536、完整H/P双流、官方full activation checkpointing、无CPU offload。启动前8卡仅4–5MiB、利用率0%，CPU quota120核、affinity200；未占用其他GPU任务。

原计划3步；第一步官方GEN MLP前向即OOM，未完成一次optimizer更新，无可报告的稳定秒/步或训练吞吐。rank1日志显示PyTorch已分配76.92GiB，GPU仅余562.56MiB，`up_proj`还需2.99GiB；rank3/6也在同类MLP临时张量处分配失败。这是容量失败，不是CPU测试或静态review已覆盖的问题，不能标为GPU验收通过。全部进程已退出（exit_code=1），之后8卡恢复4–5MiB/0%。未自动重试、恢复或启动正式训练。

完整临时证据保留远端 `tmp/ar_it2v_v04/smoke_20261005T155724Z/`：`launch_plan.json`、`Tdebug3/started.json`、`exit.json`、`logs/formal_sft.log`。本地输出的W&B ID为`qe0gsf6c`，仅作为失败短测身份，尚未用API核实，不作为正式训练结果。

只读核查的下一候选是官方sequence-sharded CP2叠加HSDP8×4，并使用官方Trainer的两次梯度累积，保持每optimizer步32个独立源pack及65K完整segment范围。需适配V0.4 memory接口、CP组内历史RNG/owner、官方CP梯度校正，并验证输出/梯度与实际显存；尚未实现或派发。官方当前没有可直接启用的MLP token chunking配置。已请求用户确认新的GPU测试，不能把候选方案写成已验证的解决办法。
