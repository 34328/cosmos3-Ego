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

2026-10-06后续决定：用户明确否决CP2，原因是此前已遇到Cosmos3 CP2问题。上述CP2候选撤销，未实现、未派发；保持CP1且禁止CPU offload。下一方向为预测块分两组分别前向/反向、累积后一次更新，保留原loss分母与噪声。尚未实现或启动新的GPU短测。

## 用户批准50K单次前向容量试验（2026-10-06）

用户为避免两次前向的历史重算开销，批准源预算从65536降为50K；按官方128对齐向下取49920。仍一次H/P并行前向、CP1、无CPU offload，loss/噪声/模型不改。新统计位于 `outputs/maintenance/full_segment_retention_49920_20261006/summary.json`，不能复用65K回执。训练32355段中保留32010段，原始时间跨度89.5189小时；29564段全部帧保留，2446段按90%至50%规则均匀保留，345段在50%仍超预算被明确排除（5.9912小时，占6.2728%原时长）。测试集1180段中预算覆盖1162段，仅统计、不混入训练。此次短测尚待GPU结果。

### 50K GPU短测结果：3步正常退出

源提交 `fc10d24`，Tdebug3单节点8×H800，HSDP8×1/CP1、单次H/P前向、官方full AC、无CPU offload。实际源预算49920，保留统计records SHA256 `799a9cd94282d266328503144baadee61361e60dc97f912a1bda55a8a93f0eae`。CPU实际配置核对确认模型、Packer、dataset和统计回执预算一致；原9项逻辑测试及Astra审查复用，无模型/loss改动。

| step | 视频loss（跨rank均值） | 训练步秒 | 峰值已分配GiB | 峰值预留GiB | 源token填充率 | 全局segment数 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.458924 | 125.73 | 67.80 | 77.61 | 96.54% | 32 |
| 2 | 0.440696 | 99.84 | 74.52 | 77.65 | 96.42% | 32 |
| 3 | 0.435033 | 98.53 | 74.51 | 77.65 | 96.76% | 24 |

最大实际Transformer序列99521 token。三步梯度范数10.8484/9.0528/9.1360，均有限并触发原梯度裁剪；无STOPPED或OOM。官方warmup实际LR为0/1e-6/2e-6；第一步LR=0，因此这里只报告3个完整训练迭代，不把它描述为3次非零参数更新。后两步monitor区间均值99.18秒，不含启动/数据预热；第一步含首次编译等开销，不能用作稳定吞吐。3步不证明长期稳定或生成质量改善，也不是四节点速度实测。

北京时间2026-10-06 08:23:08退出码0，GPU已释放。W&B API确认 [kgkw09i9](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/kgkw09i9) 为finished、step3、loss0.435033、LR2e-6。计划、日志、monitor和API回执保留 `tmp/ar_it2v_v04/smoke_50k_20261006T001220Z/`。本轮仅容量短测，无save500 checkpoint，未启动四节点正式训练。


## 时间步复用与P-only输出优化（2026-10-06）

源提交 `6370d37`。仅复用重复时间步embedding并省去H最终输出投影，训练数据、噪声、mask、loss、历史梯度、49920预算、CP1/full AC与优化器配方均不改；无CPU offload。独立Astra/xhigh静态审查未发现阻塞问题。21项CPU检查通过，2项CUDA BF16重复行梯度检查通过；CPU比较最大relative-L2约8.24e-7，CUDA约5.05e-7。开发测试和完整日志均仅在ignored tmp。

Tdebug3单节点8×H800，重放基准相同seed/Nano起点的3个训练迭代。逐条比对8个rank共24条dataloader trace一致，3步的source/Transformer token长度、segment数与LR完全一致。GPU启动前空闲；两次短测都正常退出，无OOM/STOPPED，第一步LR=0。优化短测北京时间08:47:45 exit0；W&B API核实 [ugwngy5g](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/ugwngy5g) finished、step3、loss0.435220、LR2e-6。

| 指标 | 优化前 fc10d24 | 优化后 6370d37 | 变化 |
|---|---:|---:|---:|
| 后两步平均训练耗时 | 99.18秒 | 95.79秒 | 减少3.42% |
| 全局source有效序列token/秒 | 3888.81 | 4026.42 | 增加3.54% |
| 全局H/P Transformer token/秒 | 7764.06 | 8038.79 | 增加3.54% |
| 三步峰值已分配显存 | 74.5223 GiB | 74.5060 GiB | 仅减少16.68 MiB（0.022%） |
| 三步峰值预留显存 | 77.6504 GiB | 77.6563 GiB | 基本不变 |

优化后三步分别116.96/94.34/97.25秒，loss .459064/.440851/.435220，梯度范数10.5051/8.9366/9.2104，均有限。相对基准loss绝对差1.39e-4/1.55e-4/1.87e-4；梯度范数相对差3.16%/1.28%/0.82%。数据顺序一致不等于完整BF16训练逐位等价，这些差异如实保留，不能以小模型单测替代完整运行的数值证据。

本次主要观察到少量速度收益，未解决主Transformer激活造成的显存压力；不能据此恢复65K预算。计时只有两个预热后样本，不是统计稳定的提速结论，也不是四节点通信/吞吐或生成质量评测。完整回执、API记录及对照JSON在 `tmp/ar_it2v_v04/smoke_50k_optimized_20261006T004002Z/`；未启动四节点正式训练。


针对完整运行差异，另做一次实宽、无训练的模块隔离核对：H800、TF32关闭、官方4096宽时间MLP、97920行→203个唯一时间步，FP32输出逐位相同，参数梯度relative-L2最大1.10e-5；BF16输出头4096→192、97920行→48960行，P输出和P输入梯度逐位相同，weight梯度relative-L2为2.70e-4。固定随机权重测试不代表已定位正式Nano全图的全部差异，不将3.16%范数差自动归因于TF32或称为逐位等价。证据 `tmp/ar_it2v_v04/real_width_cuda.log`，exit0；未新增训练、未更新参数。模块局部峰值下降不能与整模型峰值节省混为一谈。
