# V0.3 实验记录

2026-10-05：用户批准从官方 Nano 重新训练，单目标块 DF、GT 历史采用 LingBot-VA 公开代码噪声、窗口32 latent，四节点32卡、5000步/save500。

设计与配方见 [design.md](design.md)。CPU：17项新临时用例＋47项旧模型/配置/注意力/monitor/KV回归，共64项通过；覆盖目标分子分母、历史输入与输出两条梯度、未来隔离、部分末块、RNG恢复和官方逐latent timestep路由。新TOML也通过官方schema组合，确认窗口32、5000步、75008预算与不加载训练状态。开发测试均在 ignored tmp，不入Git。

## GPU短测

Tdebug6 单节点8×H800，官方 Nano 起点、同75008预算/local32/目标loss与噪声；仅复制数值配置为3步/replicate1，不改正式训练配方。20261005T121423Z，退出码0。三步loss=.42330/.51367/.55897，梯度均有限，正常触发clip；短测仅验收可运行和显存，不据此判断收敛。

首步73.33秒含首次计算初始化；后两步42.10/46.25秒。全局样本51/32/31，平均填充率98.44%/97.31%/97.10%；最大allocated62.249GiB/reserved71.906GiB。回执与测试脚本均在 ignored `tmp/ar_it2v_v03/smoke_20261005T121423Z/`，不入Git。

## 正式训练

2026-10-05 20:21北京时间分别经MCP启动 Tdebug2/3/4/5，各8×H800，HSDP8×4、CP1、NCCL Socket/eth0 内网TCP，master10.3.12.18:29891。从官方Nano重新开始，不从旧step1500恢复；5000步/save500。

训练源：`354f3e009813f64b2c257456001d1a06b9800ec1`；TOML SHA256 `417dc05e089ddcc82809e98878f9a3009e1116b700fb6d456a6004acd57f8475`；模型SHA256 `216264550193e143dc28c692196a28bb4eb0e0c83d1737c611bd77f521ddab3e`。

run：`target_only_df_lingbot_history_ctx32_5000_20261005T121615Z`；目录 `outputs/pretrain/ar_it2v_v0.3/20261005T121615Z/rbs_wam_ar_it2v/ar_it2v_v0_3/<run>`。启动计划及四端started/exit回执：`outputs/maintenance/ar_it2v_v03_target_only_preflight_20261005T121615Z/`；完整清单hash保存在该计划，沿用已验收的 `full_segment_retention_75008_20261003/summary.json`，records hash `631670eccc7aad65bd03f17207f621c68d0c07a04449c55e3d767ae3cf6e3b1d`。

W&B online，API确认本run为 [fv40e8uq](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/fv40e8uq)。仅step1完成，video_loss=.47314145；随后第2次反传Tdebug3/rank13(localGPU5)OOM，尝试申请1.70GiB而free1.67GiB，process77.42GiB（allocated60.53/reserved未使用14.92）。API当时只有step0元信息，没有loss/LR历史上传；不能称为正式持续训练验收。

失败批次74564 tokens，真实帧[1060,66,54,39]、latent[266,18,15,11]，均100%保留。未进入21–23步profiling，通信占比未测得。四端exit_code均1；原OOM端先退出，其他三端按本run精确launcher/PID身份清理等待进程，保留真实退出回执和 `failure_summary.json`，未自动重试。独立 GPT-6 Astra/xhigh 完整审查未发现训练语义阻塞问题，确认不能以单节点短测证明32-rank最坏批次的显存安全。

## 65536预算修订

用户随后批准65K＝65536，从官方Nano重新开run，其余配方不动。重用同一官方统计入口生成绑定新预算的完整receipt；train全帧段31266→30812（95.23%），排除22→40段，额外排除.38539小时（23.12分钟）。训练可用32315段；test额外14段进入抽帧、排除2→3段。

2026-10-05 20:55北京时间，分别经MCP启动 Tdebug1/3/4/5，各8×H800，HSDP8×4、CP1、NCCL Socket/eth0 内网TCP，master10.3.12.57:29893。Tdebug2多次连接超时且尚无启动claim，故在派发前替换为空闲Tdebug1；计算配方不变。新run为 `target_only_df_lingbot_history_ctx32_65k_5000_20261005T124732Z`，输出 `outputs/pretrain/ar_it2v_v0.3/20261005T124732Z/rbs_wam_ar_it2v/ar_it2v_v0_3/<run>`，四端独立回执在 `outputs/maintenance/ar_it2v_v03_target_only_65k_preflight_20261005T124732Z/`。

训练源 `20f717bf8c8663c569d0bd610fa25585900077ec`；TOML SHA256 `ef6cda600b004d6c98157118ba5731a6725f207e509d4acf9f7a82020c95c850`。模型与清单hash同上；新统计回执 `full_segment_retention_65536_20261005/summary.json`，records SHA256 `22697620df91aee713dcae1a454dded5d0613f759ddfaa03316050a3ac898967`。Astra/xhigh复核了新配置、预算回执及Nano起点，未发现阻塞问题。

21:03北京时间，真实monitor已完成step5，四端均无退出；W&B API核实新run [daxa2qx7](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/daxa2qx7) online/running，history step2/3/4含video_loss=.48592/.51343/.46096及LR=1e-6/2e-6/3e-6，证明持续上传，API回执保存在preflight目录。step3–5耗时40.29/39.61/38.67秒，global样本107/131/115，token平均填充96.88%/96.61%/96.62%；最大allocated56.95GiB/reserved77.13GiB，梯度有限、无STOPPED。此为启动阶段实测，不代表完整5000步的最坏显存或收敛验收。

21:19北京时间，monitor/API均核实到step28（loss=.20549、LR=2.7e-5），四端无退出或STOPPED。排除初始两步及采样/导出21–24步，step3–20/25–28共22步平均40.63秒（38.19–44.12），平均每步108.59个segment、填充96.70%；累计最大allocated56.96GiB/reserved77.26GiB。`startup_acceptance.json`保存在preflight目录。按当前速度5000步约56.4小时，不含周期性checkpoint保存等额外开销。

## 一次性通信测量

复用官方profiler，在本次正式run的21–23步采样rank0/8/16/24，代表四节点同一个replicate lane，不代表全部32rank的平均。临时wrapper、4份完整trace、wall回执和CPU汇总均在 ignored `tmp/ar_it2v_v03/comm_target_only_df_lingbot_history_ctx32_65k_5000_20261005T124732Z/`，不入Git；GPU采样已自动结束。Tdebug6仅CPU聚合，退出码0、51秒。

按实际process-group和trace关联区分跨节点与单节点，全部NCCL kernel均归属成功、unknown=0；跨机replicate PG359=[0,8,16,24]，本机shard PG367/368/369/370分别含各节点8卡。每个rank在每步内合并通信区间，除以真实wall时间，不累加重叠kernel或跨rank时间。

| 采样节点/rank | 平均wall秒/步 | 跨节点活动区间占比 | 未与计算/拷贝重叠秒/步 | 未重叠占比 |
| --- | ---: | ---: | ---: | ---: |
| Tdebug1/0 | 41.12 | 32.65% | 3.48 | 8.45% |
| Tdebug3/8 | 41.12 | 44.72% | 3.19 | 7.77% |
| Tdebug4/16 | 41.12 | 54.67% | 4.10 | 9.97% |
| Tdebug5/24 | 41.12 | 31.53% | 3.52 | 8.56% |

跨节点活动有72.85–82.63%与本卡计算/拷贝重叠。未重叠部分是暴露的通信活动估计，含等待其他rank，仍可能与其他通信重叠；不能解释成纯网络传输耗时或可消除的加速比例。原始32–55%活动占比不能叫额外通信开销。采样仅三步，不是单/双/四节点同工作量的扩展效率对照；本次通信临时测量到此结束。

## 用户终止单目标实验

2026-10-05 23:14 北京时间，按用户要求停止本次65K单目标实验，以改为历史条件副本与所有预测块并行监督。分别核对四端本run的torchrun PID/birth tick后发送SIGINT；四端退出码均1，属于用户中断，不能标作数值失败或训练完成。各节点的 `user_stop_request.json` 与 `exit.json` 留在上述65K preflight目录。

最后完整步为194：video_loss=.18353061，原始梯度范数=.19859269，LR=9.9937801e-5，41.31秒/步，global batch127，平均源token63782.125（97.32%填充），本步最大allocated56.89GiB/reserved77.05GiB。尚未达到save500，没有本run的checkpoint；日志与W&B历史保留。新方法另立版本，不从这次run自动恢复。
