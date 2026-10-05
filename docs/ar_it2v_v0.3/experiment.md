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

用户随后批准65K＝65536，从官方Nano重新开run，其余配方不动。重用同一官方统计入口生成绑定新预算的完整receipt；train全帧段31266→30812（95.23%），排除22→40段，额外排除.38539小时（23.12分钟）。训练可用32315段；test额外14段进入抽帧、排除2→3段。新记录待正式启动与API确认后填写。
