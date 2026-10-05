# V0.3：单目标块 DF 与 GT 历史

用户于2026-10-05批准：从官方 Cosmos3-Nano 权重重新开始，四节点32卡训练5000步，每500步保存。目标首先是提高 GT history 下的单块预测质量；不声称这次已解决自由生成误差累积。

## 训练时序

一个完整 segment 连续 VAE 编码，保留原文本与段边界。分块仍为 `[1,4,4,…]`，最后一块可不足4 latent，补帧继续使用原真实帧权重。

每个样本均匀选一个非条件目标块 k。首 latent0 是唯一干净图像条件。目标 k 使用原 V0.2 块内共享 σ：uniform 抽样、shift5、clamp[.02,.98]；速度目标仍为 ε−x0。

目标 k 前的 GT 历史采用用户选定的 **LingBot-VA 公开代码**口径：每个样本50%概率整个历史保持干净，否则每个历史 latent 独立均匀抽取 timestep index500..999。官方1000步 FlowMatch scheduler 的 descending σ 经 shift5 后，历史 σ 最小约.004975、最大.833333，中位数约.625。没有再次 clamp 到目标σ的范围。首次图像条件恒为σ0，其他σ0历史仍是普通历史 token，不标为条件帧。

注意力总窗口从16增加至32 latent，即当前4 latent＋最多7个历史块；块内双向、块间因果，无永久首帧 sink。维持样本隔离和绝对位置。k之后的块保留原输入但对目标不可见，不参与 loss。整段单遍前向，历史隐藏 KV 不 detach，因此目标梯度可回传到历史表示。

只有目标 k 进入 loss 分子及有效分母；复用原生 flow loss、真实末帧权重及每样本平均。历史块预测的速度无直接监督。不是旧联合 V0.3.1 的“只改分子、保留全段分母”配方。

原整段目标σ抽样先执行，目标选择和历史噪声使用独立、按 seed/iteration/rank 派生的 CPU RNG；不推进原 σ/ε 的 RNG。选择元数据仅属于本次 packed sequence，training_step 成功或异常后均清除。恢复沿用官方 DCP 和原可恢复 Packer，不另写训练循环。

## 与 LingBot 的边界

对齐的是公开代码的历史噪声分布和每 latent 粒度，不是复现整个 LingBot 架构、数据或双流 Transformer。该论文描述50%干净/50%σ∼U(0,.5)，而公开实现经过 shift5 后分布不同；用户明确选公开实现。来源固定为 [LingBot-VA commit 7c6ffa9](https://github.com/Robbyant/lingbot-va/tree/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb)：`wan_va/train.py` 的 `_add_noise`、`utils.py` 的 timestep index 采样、`scheduler.py` 及 robotwin 配置 shift5。论文：[LingBot-VA](https://arxiv.org/html/2601.21998v1)。

## 保持的配方

- 官方 Nano 初始化、GEN训练/UND冻结、纯视频 IT2V；无 action。
- 完整连续原始 segment、65536 token、官方动态 packing；超预算段沿用既有批准90%→50%保留策略，按新的预算重算回执。
- 官方 Trainer/VAE/flow/AdamW/调度/DCP；LR峰值1e-4、warmup100、LambdaCosine cycle5000/f_min.3，wd.01、clip1、seed42。
- HSDP8×4、CP1，NCCL内网Socket/eth0 TCP；W&B online，API核实。
- 原数值/OOM/内存与10步loss停止规则保留。失败不得自动重训或恢复，不因主观画质停训。

入口：`cosmos3_ar_it2v/model_v03.py`、`config_v03.py`、`configs/ego100h_target_only_df.toml`。旧 V0.1/V0.2 模型和配置不变。

开发测试和一次性通信 profiler 仅在 ignored `tmp/ar_it2v_v03/`，不进Git。复用官方 profiler，在正式训练21–23步采样四个跨节点代表 rank，随后关闭。报告同机/跨节点NCCL活动区间并集及与计算重叠，不能由 kernel 耗时求和断言纯通信开销。

2026-10-05预算修订：初次75008/local32正式run在第2步反传OOM，用户批准改为65536并重开，不更改模型/窗口/噪声/loss/优化器。新版数据回执 `outputs/maintenance/full_segment_retention_65536_20261005/summary.json`：train32355段中30812全帧保留，32315段可用，40段50%仍超限被排除；相比75008，454段进入额外抽帧、排除增加18段（原时长增加.38539小时）。短段不裁剪，整段时间跨度和文本保持。
