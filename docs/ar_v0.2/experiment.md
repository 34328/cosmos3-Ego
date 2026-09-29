# V0.2 数据、训练与验收记录

> 2026-09-30 清理说明：按用户要求，已删除训练前短测的 checkpoint、原始日志、trace、临时文件及12个在线W&B测试run和本地副本。下文测试结论为清理前的真实验收记录，所列旧测试输出路径不再代表文件仍保留。PCA及其来源记录、数据清单、当前默认配方的归一化统计、已有正式训练600/1200步checkpoint和回归测试源码保留。后续按用户明确要求，prepared257/273两套测试统计也已删除；需要复用长档时须重新准备并校验对应统计，不可直接沿用已删除的路径。清理清单及关键验收汇总见远端 `outputs/maintenance/cleanup_pretrain_20260930.json`。

> 2026-09-29 早先记录：当时方案切换为 `fixed_camera_delta_latent_v1`，训练与推理固定 C=4、K=8，每个完整块 32 条 action。旧实验、旧 AE 和旧统计保留原配置与真实结论，但不能用于新表示。

> 2026-09-28 配方更新：下一轮 V0.2 使用 `L_total = L_video + L_action`，action 系数从 0.7 改为 1.0。已完成训练、短测及其 checkpoint/日志仍保留当时的 0.7，不按新系数重写历史。

系数更新验证：loss/日志测试首轮其余 74 项通过；旧目标常量 3.8 更新为新目标 5.0 后，训练合同和三份官方 TOML 配置加载复验通过。断言覆盖实际总 loss、梯度和生效 action 系数 1.0。日志在 `outputs/maintenance/action_weight_1_regression_20260928{,_retry}.log`；未启动新的 GPU 训练。

更新：2026-09-29。方案以 [design.md](design.md) 为准；本文把当前实施状态与历史运行分开，历史失败和通过记录均按原条件保留。以下产物路径均相对远端 `/mnt/lzh/cosmos-EgoWAM/`，不是本地文件下载链接。

## 0. 当前新表示：wrist-local PCA15＋Δz

### 当前：训练前第5步，双节点16卡短测通过（2026-09-29，待Claude复审）

仅短测，未启动正式训练。Tdebug2（10.3.12.18，MASTER）和Tdebug6（10.3.12.24）各8×H800；每轮启动前两台均8卡空闲，结束后也已释放。官方HSDP shard=8、replicate=2；实际trace核实节点内分片组[0…7]/[8…15]、跨节点复制组[0,8]。两端分别经MCP调用 `launch_ar_v0_2.sh`，NNODES=2、NODE_RANK=0/1、MASTER_ADDR=10.3.12.18。只设置NCCL_SOCKET_IFNAME=eth0与NCCL_IB_DISABLE=1，无其他NCCL调优。

配方为fixed-camera，cuDNN benchmark=false、deterministic=true；273混合档 `[273,257,129,65,33]`，C=4、60K token、最多4clip/卡、grad_accum=1。复用第3步已校验且hash匹配的prepared273统计／清单，没有重新拟合。独立证据目录：`outputs/validation/ar_v02_pretrain_step5_20260929T233700/`。`launch_nodes.py`只是本次参数与进程启动记录，底层复用官方CLI／Trainer／checkpoint／W&B，没有新增训练循环或修改模型源码。

**恢复验收通过：**不中断基准1～6步，第4步保存；将其完整checkpoint4硬链接复制到独立replay目录，从4恢复仅跑5、6。两阶段两节点退出码均0、无OOM；checkpoint4及两轮checkpoint6的模型／optim／scheduler／trainer metadata和全部16rank数据状态均核验通过。16rank×2步data trace、源帧／文本／action／state hash、实际双路sigma完全一致；全部loss最大绝对差0，断言仍为atol=1e-6、rtol=1e-5。比较器显式传ranks=16，未沿用CLI默认8。视频RGB像素本体未单独hash。证据为`replay_comparison.json`、`checkpoint4_verified.json`及`*_checkpoint6_verified.json`。

**在线记录：**16卡基准（测试run `50q9nhk3`，2026-09-30已删除）和16卡恢复（测试run `x4sb2149`，2026-09-30已删除）均online、API核实finished。分别逐步核对1～6和5～6的17项loss/sigma，与本地JSONL一致，含总loss、video/action整体及8项action字段。验证结果在`wandb_reference_verified.json`和`wandb_replay_verified.json`。

基准各步计时如下。同步训练段取16rank最大值，含前向／反向／优化器，不含取数、保存和步后日志；官方整步时间直接取IterSpeed。首步含初始化／编译，第4步整步含保存；最终第6步后的收尾保存不在第6步IterSpeed内。

| step | 同步训练段 s | 官方整步 s | global batch（实际clip总数） |
|---|---:|---:|---:|
| 1 | 37.17 | 91.16 | 57 |
| 2 | 29.46 | 33.38 | 51 |
| 3 | 27.61 | 33.61 | 46 |
| 4 | 27.82 | 53.03 | 48 |
| 5 | 24.82 | 26.24 | 40 |
| 6 | 28.56 | 30.58 | 38 |

每rank实际clip数和峰值如下；rank0～7在Tdebug2，8～15在Tdebug6。恢复5/6各rank的clip数与表内基准5/6相同。GiB为Torch allocated/reserved，含缓存，不等同nvidia-smi进程总显存。

| rank | 基准step1→6 clip数 | 基准峰值allocated / reserved GiB | 恢复峰值allocated / reserved GiB |
|---|---|---:|---:|
| 0 | 3/4/4/3/2/2 | 54.70 / 64.82 | 53.43 / 63.67 |
| 1 | 2/3/3/4/2/2 | 55.61 / 66.30 | 53.43 / 63.72 |
| 2 | 3/4/3/2/2/2 | 55.62 / 66.35 | 53.44 / 63.89 |
| 3 | 4/3/4/3/2/3 | 55.61 / 66.74 | 53.44 / 63.89 |
| 4 | 4/3/4/3/3/2 | 55.62 / 66.84 | 55.62 / 65.89 |
| 5 | 4/3/2/3/2/2 | 56.43 / 67.27 | 53.43 / 63.32 |
| 6 | 4/3/2/2/4/2 | 55.51 / 66.51 | 54.71 / 65.27 |
| 7 | 3/2/1/3/1/1 | 55.61 / 66.30 | 49.26 / 58.78 |
| 8 | 4/4/3/3/2/4 | 55.61 / 67.10 | 52.68 / 62.50 |
| 9 | 3/3/3/3/3/2 | 55.62 / 66.93 | 53.43 / 63.70 |
| 10 | 4/4/4/3/4/3 | 55.61 / 66.52 | 55.61 / 66.81 |
| 11 | 4/4/3/3/4/3 | 55.62 / 67.21 | 55.51 / 66.38 |
| 12 | 4/2/3/2/2/4 | 55.61 / 66.44 | 54.69 / 64.58 |
| 13 | 3/3/2/3/2/2 | 55.62 / 66.29 | 51.40 / 61.05 |
| 14 | 4/3/3/4/3/3 | 55.62 / 66.67 | 55.62 / 66.47 |
| 15 | 4/3/2/4/2/1 | 54.60 / 65.01 | 53.43 / 63.78 |

基准峰值allocated **56.43GiB**、reserved **67.27GiB**；恢复55.62/66.81GiB。基准总消费280clip，平均46.67clip/步，global batch随packing变化，不能按“16卡×4”固定算64。所有96个rank-step均未超过60K／4clip，实际帧数组见`reference_summary.json`。恢复同步段5/6为37.25/25.02秒，官方整步120.23/29.34秒；恢复首步包含加载／数据队列重建／编译，且本轮开启官方profiler，不能拿它当稳定性能基准。

**与单节点273档及跨节点开销分别比较：**单节点记录来自同型号Tdebug2、第3步273混合档，配置／统计／确定性设置相同。仅有3个样本，且分片后的数据组合不同，不当成严格同工作量性能消融。

| step | 单节点同步段 s | 双节点同步段 s | 增加 s | 单节点→双节点clip总数 | 单节点→双节点官方整步 s |
|---|---:|---:|---:|---|---|
| 1（启动） | 37.80 | 37.17 | -0.63 | 28→57 | 143.99→91.16 |
| 2 | 27.49 | 29.46 | +1.97 | 26→51 | 34.18→33.38 |
| 3 | 26.22 | 27.61 | +1.40 | 24→46 | 51.97→33.61 |

第3步单节点含checkpoint而双节点不含，整步时间不可直接相减称为加速；第2/3步同步段增加7.15%/5.32%，这是实测净变化，包含负载差异，不能全部归因通信。

为单独识别通信，恢复轮使用**官方profiler**，第5步预热、第6步采集，导出rank0/8；trace标签`ProfilerStep#5`为零起始计数，对应优化器第6步。未改loss、NCCL参数或验收阈值。trace内按真实process group识别跨节点复制通信：

| 第6步profile | 跨节点replicate all-reduce次数 | replicate核持续时间合计 s | 未与非NCCL GPU计算核重叠的部分 s | 全16rank默认组通信核合计 s |
|---|---:|---:|---:|---:|
| rank0 | 38 | 7.940 | 0.842 | 0.505 |
| rank8 | 38 | 2.922 | 0.388 | 0.684 |

这是两张代表卡的直接通信证据，不是16rank平均或整步额外延迟。通信核包括等待对端，且可与计算／其他通信重叠；两卡数值不同不能解释成网络速度不同。“未与计算重叠”仍不能证明全部位于关键路径。因此不把7.940秒直接加到基准整步，也不伪造唯一的纯通信开销。可观测的是上述通信占用，以及无profile基准相对单节点的+1.40～1.97秒净变化。原trace在replay运行目录`torch_trace/iteration_6/`，汇总在`communication_profile_summary.json`。

**保留失败记录：**第一次启动把Hydra路径写成`model.parallelism`，两节点在配置组合阶段退出1，未进入模型或创建W&B run。按实际schema改为`model.config.parallelism`，先CPU验证组合配置再启动；失败目录`failed_config_reference/`和旧启动参数快照保留。这是本次启动参数错误，不是HSDP或网络故障。

本次结论仅覆盖当前16卡、混合档的6步及5/6恢复；长期稳定性和训练质量不在通过范围。正式训练仍须用户指定训练量与保存点，不能因短测通过自动启动。

### 前序：训练前第3、4步短测完成（2026-09-29）

仅验证，不启动正式训练。129/257档在Tdebug6、273档在Tdebug2，各自单节点8×H800 80GB，复用 `launch_ar_v0_2.sh`→官方CLI／Trainer／DCP／W&B；固定C=4、60,000 token、最多4clip、grad_accum=1。独立证据目录为 `outputs/validation/ar_v02_pretrain_steps34_20260929T215824/`。测量callback默认关闭，仅本次短测开启，不另写训练循环。

129帧基准6步已通过，退出0，无OOM。各卡step1～6实际clip数和整轮峰值如下（GiB，allocated与reserved分开）：

| rank | clip数（step1→6） | 峰值allocated | 峰值reserved |
|---|---|---:|---:|
| 0 | 2/2/3/2/2/2 | 56.45 | 66.04 |
| 1 | 2/2/2/2/2/2 | 48.34 | 57.92 |
| 2 | 2/2/2/3/2/2 | 56.44 | 65.14 |
| 3 | 2/2/2/2/2/2 | 48.33 | 57.49 |
| 4 | 3/2/2/3/2/2 | 56.44 | 65.95 |
| 5 | 2/2/2/2/2/2 | 48.32 | 57.60 |
| 6 | 2/2/2/2/2/2 | 48.33 | 57.57 |
| 7 | 2/2/2/2/2/2 | 48.34 | 57.76 |

每步取8卡最大同步训练步耗时：64.52（首步含编译）、13.95、20.09、20.51、13.95、13.95秒。后5步中位13.95秒；这段计时含前向／反向／优化器，不含取数、checkpoint及步后日志，不能称为完整端到端吞吐。129基准是旧cuDNN自动选核设置的实测，后续确定性设置另行报告，不混成同一组。

按字段action日志已经接通：相机平移／旋转、右腕平移／旋转／手形、左腕平移／旋转／手形，共8项；见design §5.2。整体loss、梯度与RNG不变，逐样本等权；不是重新按8项求和。真实GPU日志按字段维数加权还原整体action，最大差约1.81e-8。129帧基准（测试run `p93kic1u`，2026-09-30已删除）已通过API核对6步、17项loss/sigma与本地JSONL一致。

保存恢复采用基准1～6步，在第4步保存；新目录仅复制该checkpoint，再恢复第5、6步，与不中断基准对照。首次恢复的数据hash、sigma一致，但部分loss差1e-4～1e-3，超过预先固定的 `atol=1e-6, rtol=1e-5`，**该次失败保留**，没有放宽阈值。cuDNN从 `benchmark=true, deterministic=false` 改为 `false,true` 后重复完整基准／恢复，两轮8卡第5、6步的data trace、源帧／文本／action／state hash、实际双路sigma与所有loss均完全一致，最大差0。没有对视频像素本体做hash，也不将本次8-rank结论外推到16-rank或其他环境。对照支持cuDNN选核是本次恢复差异的来源，没有进一步定位到某个卷积算子。

确定性基准（测试run `lsj8o51h`，2026-09-30已删除）和确定性恢复（测试run `owqzvq8r`，2026-09-30已删除）均正常退出，W&B API核实finished且17项指标逐步与本地一致。新fixed-camera配方固定上述cuDNN设置；legacy不变。`deterministic_replay_comparison.json`保存逐rank断言结果。

确定性129基准各rank的allocated峰值与上表一致；reserved峰值依次65.26/57.62/65.49/57.73/65.23/57.14/57.36/57.74GiB。各步同步耗时26.06/14.02/20.15/20.61/14.01/14.03秒，后5步中位14.03秒；首步编译缓存状态不同，不据此宣称确定性设置加速。

官方IterSpeed整步墙钟另报：旧cuDNN基准step1～6为235.74/18.31/21.22/47.87/15.31/15.55秒；确定性基准为82.13/19.84/21.54/46.94/15.66/15.56秒。首步含启动／编译，step4包含checkpoint；普通步和保存步不混合求“稳定吞吐”。这些不是上文仅同步训练段的14～20秒。

257/273档的独立审计清单和两套统计均已完成、exit0，未覆盖原正式统计。257包state/future行数为277,424／8,877,568；273包为286,432／9,165,824，各35,272个train窗口。hash与往返断言保存在各自`prepared257/summary.json`、`prepared273/summary.json`。

**资源使用失误与修正：**本次两套全量准备各单核计算约45分钟；Tdebug2实查有约120核CPU配额和1.4TiB可用内存，未提前利用是执行安排失误。补充`--workers`多进程编码：主进程固定seed抽样，worker不抽样，按原顺序收集；每worker限制1个Torch线程，Linux fork共享只读源数组与codec；增加完成窗口数／吞吐日志。21项回归通过，串行和2worker生成的清单、采样、两套统计及summary逐字节一致。原任务恰好完成，未为展示并行重新跑两遍全库；没有声称32worker全库已经实测。未来启动须显式按空闲配额选并发，CLI的1worker默认保留作串行参考。教训已写入根AGENTS.md并同步。GPU两档利用空闲同型号节点独立运行，不是双节点训练，耗时比较存在跨节点差异。

257档已完成3步，无OOM。在线run（测试run `hcrmmuwl`，2026-09-30已删除）通过API核对3步17项指标。273档也完成3步、无OOM，两档均exit0；完整模型／optim／scheduler／trainer及8rank数据状态的checkpoint核验通过。预算始终60K、最多4clip。正式训练和双节点训练未启动。

| 257档rank | clip数（step1→3） | 峰值allocated GiB | 峰值reserved GiB |
|---|---|---:|---:|
| 0 | 3/4/4 | 54.71 | 64.27 |
| 1 | 3/3/2 | 52.52 | 61.42 |
| 2 | 3/3/4 | 54.59 | 63.82 |
| 3 | 4/3/4 | 54.58 | 63.41 |
| 4 | 3/2/2 | 56.43 | 65.49 |
| 5 | 4/3/3 | 54.59 | 63.88 |
| 6 | 4/4/2 | 52.54 | 61.71 |
| 7 | 2/2/4 | 52.63 | 61.75 |

257实际混合`[257,129,65,33]`，所有rank在三步内都消费了257帧clip；不是仅配置长档而未实际取到。完整每卡每步帧数组在`memory257_summary.json`。同步训练段34.39/27.86/27.03秒；官方整步80.54/31.57/53.29秒（第3步含保存）。短测不能证明长时间训练不会OOM。

273档实际混合`[273,257,129,65,33]`，所有rank都消费了273帧clip；其中第17块覆盖H=15历史的首次淘汰场景。每rank帧数组保存在`memory273_summary.json`。短测确认能运行，不替代KV cache数值验收或训练质量评估。

| 273档rank | clip数（step1→3） | 峰值allocated GiB | 峰值reserved GiB |
|---|---|---:|---:|
| 0 | 3/4/4 | 54.71 | 63.63 |
| 1 | 3/3/2 | 52.52 | 61.83 |
| 2 | 3/3/4 | 55.53 | 64.63 |
| 3 | 4/3/4 | 55.61 | 64.29 |
| 4 | 3/4/3 | 55.52 | 64.41 |
| 5 | 4/3/3 | 55.63 | 65.26 |
| 6 | 4/4/2 | 53.43 | 62.45 |
| 7 | 4/2/2 | 53.44 | 62.64 |

同步训练段37.80/27.49/26.22秒；官方整步143.99/34.18/51.97秒（第3步含保存）。273帧在线run（测试run `lpsbtvtt`，2026-09-30已删除）经API核实finished，3步17项loss/sigma与本地JSONL一致。两档样本组合不同、节点不同，273峰值略低不能解释成更长clip更省显存。以上为3步容量短测，没有做长期稳定性承诺。

**本轮结论：**第3步三个档位均短测通过；第4步八项字段日志、本地与在线一致性、官方保存退出恢复均通过，其中精确恢复依赖已固定的cuDNN设置。8卡129帧恢复结论不外推到16卡；长档仅验证容量和保存，没有另做长档恢复对照。下一步由Claude Code复审，正式训练仍需用户另行指定。测试与修改清单见review §7.10；完整本轮差异和hash在证据目录`implementation.patch`、`changes.json`。

### 当前：PCA15 拟合与119个heldout验收（2026-09-29）

用户取消MLP AE重训，改为左右手各自拟合PCA15。实际未启动MLP或WAM训练；在Tdebug4用CPU完成FP64协方差分解，输入为米制 `q=R_wristᵀ(kp−p_wrist)` 展平60D，只中心化，15D编码再标准化。固定seed=42，无随机SVD、无超参搜索、无heldout调参。

当前744个train episode全部覆盖，每侧 **1,827,090** 条去重手形；119个heldout episode全部覆盖，每侧 **287,891** 条。范围为冻结审计窗口覆盖的全部源帧（含边界），不声称覆盖episode中未纳入当前任务片段的其他帧。不可见但跟踪有效的手部保留；没有以FOV筛掉困难样本。按源帧去重，不把重叠窗口重复曝光当独立样本。

| 手侧 | 解释方差 | mean | P50 | P90 | P95 | P99 | max | 验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 右手 | 97.1719% | 2.951 mm | 2.535 | 5.359 | 6.533 | 9.656 | 48.975 | 通过 |
| 左手 | 97.6693% | 2.792 mm | 2.405 | 5.201 | 6.319 | 9.037 | 31.780 | 通过 |

误差口径：每个非手腕关键点的欧氏距离，mm，20点×全部heldout有效源帧汇总；两侧分别按mean≤5mm、P95≤15mm判定，未放宽门槛。另存逐episode结果、逐帧平均误差分布及直方图。每侧1024个固定z重复decode10次，最大差异为0；生产路径100个完整C=4块＋8帧尾块的reanchor检查通过，z原值继承、无重编码，物理点最大误差约1.19e-7m。

“通过”指两侧各自的全heldout汇总门槛，不表示每个episode都达标：右手有3/119个episode的mean>5mm、1/119个P95>15mm，最差episode mean=7.737mm、P95=23.078mm；左手两项均0/119超标，最差mean=4.717mm、P95=9.782mm。未据此删数据或更换权重。按逐帧20点平均误差计算的P95另为右5.062mm／左4.639mm，不与逐关键点P95混用。

产物：`cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v3_wrist_local_pca15_train744/`，包含左右`*_pca15.pt`、`*.validation.json`、详细`*.evaluation.json`和`manifest.json`。manifest／权重记录完整episode清单、源CSV／窗口／数据hash、解释方差和seed；sidecar绑定对应权重hash。v2_4未覆盖，4份MLP权重哈希与旧manifest一致。

| 产物 | SHA256 |
|---|---|
| right_pca15.pt | `7b32b7c69f1ce7e3cfc278296b382e1eedc312c9c555e76eca3622b5d9a78174` |
| left_pca15.pt | `9f0fd9219e2bc9e0a312e8445fd84193652989df3fa978cd12eb0e70686a646e` |

在线记录：[本次PCA拟合与验收](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/5sdqm350)。已通过W&B API核实finished、passed=true及两侧真实mean/P95历史，不是仅生成URL。该run只记录PCA拟合／验收，不是WAM训练run。

Runtime增加显式PCA architecture分派，编码 `(x−mean)·Cᵀ` 再按train latent标准差缩放，解码执行逆标准化和PCA重建；历史MLP入口仍保持独立。`config.py` 的 `fixed_hand_codecs` 已指向上述目录；官方训练preflight支持codec与统计分目录。

两套57D统计在PCA通过后启动。首次运行在现有FP32通用4×4求逆处失败：有效刚体位姿产生约1.19e-7的底行误差，触发原1e-7检查。修复为经严格SO(3)校验后的解析刚体逆（Rᵀ、−Rᵀp），底行精确保留；不放宽跟踪／几何／质量门槛。失败产物和日志保存在`outputs/maintenance/wrist_local_pca15_20260929/failed_statistics_geometry/`及`normalizers_final.log`，修复后重新拟合。

**两套统计已完成并验证通过**：输出 `outputs/data_expansion_20260928/prepared_fixed_camera_wrist_local_delta_latent_v1/`。沿用当前配方，seed=42、每train segment抽8窗，共35,272窗；state 200,496行、future 6,415,872行。state与future分别拟合57D Piecewise-Asinh；PCA自身标准化不替代这两套统计。只用train拟合，全部拟合行通过原FP32往返断言，process exit=0；验证集未参与拟合。

| 统计文件 | SHA256 |
|---|---|
| chunk_state_normalizer.json | `fe11207c3aa0bb94933d5c90908fde45c6af228e25f9fd84c85ca4a2d2d05bbb` |
| future_normalizer.json | `5a7e28f8d3eb687e4e436e980eff010cc262e4bc7e6ba90b0268080733087038` |

最终实物校验 `artifact_verification.json` 通过：119个heldout episode各取一个完整审计窗口，检查740条state与23,680条future；57D物理表示中的最大往返误差分别9.54e-7／4.77e-7（各字段单位不同，不能将该总体数值称为mm），沿用`atol=1e-5, rtol=1e-5`。逐字段误差、归一化分位数和floor命中记录在报告。原审计窗口和源清单hash未改变；codec原件与prepared快照逐字节一致；两套统计和config绑定正确。解析逆修复后又用真实PCA权重复验100块＋尾块，最大点误差右1.79e-7m／左1.49e-7m，仍通过，见产物目录`runtime_recheck.json`。

测试：PCA／数据专项67项通过；扩展联测首轮123通过、8失败，原因是旧测试仍替换MLP类名，没有替换新的通用loader；更新测试替身入口后 **131 passed（33.55s）**。几何修复后含真实位姿回归与长链／数据检查 **66 passed（27.84s）**。集合有重叠，不相加。全部为CPU检查，未运行WAM。证据目录 `outputs/maintenance/wrist_local_pca15_20260929/`。

### 较早实施记录：切换PCA之前

2026-09-29 用户确认并授权实现 `fixed_camera_wrist_local_delta_latent_v1`：刚体固定块首相机轴逐帧增量不变；手形 `q=R_wrist^T(kp-p_wrist)`，future=Δz，关键点 `p_wrist+R_wrist·D(z)`；换块 z 原值继承。具体公式与实施边界以 design 为准。

本轮综合回归 **441 passed（130.38s，0 failed）**，覆盖几何、codec／来源、数据、归一化、模型、packer、缓存／流式推理、原始GT评测／投影、配置、checkpoint合同和启动检查。100块／3200帧FP32–FP64对照及末尾不满块均纳入；CPU合同检查不代表真实GPU生命周期通过。首次联测424 passed／11 failed，失败在新增collector使用`len(zarr.Array)`；改为`shape[0]`后通过，保留首轮失败日志。

原始GT hand MPJPE已实现并回归：预测latent等于GT latent时，辅助decoded-GT误差可为0，主指标仍包含AE重建误差。Dataset归档原始世界系关键点和显式源帧，generated不读后续GT state重置。新表示缺原始GT时主评测／投影失败，不偷偷退回解码GT。

官方薄入口为 `cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_2.sh`，复用 `_sft_launcher_common.sh` 和官方CLI。实际只读检查退出码2，`status=blocked`、`lifecycle_verified=false`：尚缺已验证AE、sidecar与两套统计。验收入口要求全新输出目录，真实跑step4→6后检查完整DCP、trainer iteration、各rank数据进度及恢复日志；本轮未执行GPU阶段。旧数值smoke仅保留算子验证用途。

Tokenizer核对：DCP声明Qwen3-VL-8B-Instruct；基础词表151643、含新增token为151669、模型容量151936，特殊token ID与Qwen3-VL配置一致，embedding/lm_head为151936×4096；词表hash与同一Nano发布目录一致。DCP没有不可变revision／tokenizer源hash，因此确认结构与token语义兼容，不声称完全同一上游revision。

真实数据审计（不是模型质量或显存验收）：

| 模型视频帧数 | train可用片段／窗口 | held-out可用片段／窗口 | 历史覆盖 |
|---|---:|---:|---|
| 当前129／65／33 | 4409／1025215 | 642／166675 | 最多7个历史chunk |
| 257单档诊断 | 1202／498876 | 190／85439 | 第16块读取15个历史chunk |
| 273单档诊断 | 1126／461355 | 173／79564 | 第17块首次淘汰 |

当前档位覆盖744 train／119 held-out episode，未因现有跟踪有效性规则额外排除窗口；这是有限值、姿态和缺标检查，不保证全部标注质量。FOV缺口按保留未来帧覆盖并集统计，最长train右／左155／104帧、held-out54／87帧；少数长缺口不能用平均可见率掩盖。此前“缺失增量影响恢复可见后的积分”测试说明了旧屏蔽策略的风险。现按用户决定采用方案 A：仅新 wrist-local 表示取消 FOV loss 屏蔽，跟踪有效的全部 future 57D 计入分子和分母，跟踪无效仍由 `invalid_frames` 拒绝整窗。`hand_visibility` 保留供审计／可选分组评测；legacy 不变。此修改不等于预测增量不再有积分误差。长clip未启用，GPU显存未测。

AE bootstrap窗口清单已生成：`outputs/maintenance/wrist_local_20260929/codec_source_audit/valid_windows.json`，schema为`ar_v02_codec_source_windows_v1`；不能作为最终训练清单。清单SHA为`5e3dccb0c50bcb33aa725217c83bf732c2e202549cdf716dcea1213588c241eb`。真实train／held-out各1个episode的collector也已通过。旧v2_4 AE缺episode来源，未批准直接复用；**新AE尚未训练和验收，新state/future统计尚未拟合**。数据准备现会将已验证AE与sidecar按原字节打包到最终统计目录并核对hash，与配置资产路径一致。

本轮证据目录：`outputs/maintenance/wrist_local_20260929/`，包括`final_integration.log`、`final_integration_exit.json`、`native_preflight.log`／`.exit`及各audit目录；`before/`保留修改前快照。未启动AE或WAM训练；真实GPU保存恢复、C=4多样本显存及双节点生命周期仍待验收。已确认配方不变。

启动／配置／GC最终增量复核16项通过（23.21s），证据`final_launcher.log`；与441项集合有重叠，不合计。

**方案 A 增量修复（2026-09-29）**：`loss.py` 增加默认保持旧行为的 `mask_out_of_fov` 参数；`model.py` 的两个整体 action loss 入口仅在表示恰为 `fixed_camera_wrist_local_delta_latent_v1` 时关闭 FOV 屏蔽。没有重写数据中的可见性值，没有修改 `invalid_frames`、loss 系数、AE、归一化或训练配方。可见／不可见分组评测可使用保留字段另行统计，本轮未新增评测分组输出。

- 专项CPU回归：**58 passed（19.24s）**，包含新增 `test_action_fov_policy.py`、原有两套 loss 回归及数据流水线。断言不可见行的分子／分母／梯度，两个模型 loss 入口的新／legacy／未声明表示分支，C=4 完整块及末尾1／2／3组，条件／结构padding不监督；数据保留FOV全0的64条future，损坏其中一帧则拒绝整窗。
- 集成CPU回归：**104 passed（53.82s）**，覆盖模型、训练合同（含两进程CPU分布式归约）、数据和可见性审计。与上组文件不重叠，本轮共 **162 passed，0 failed**；不与此前441项重复累加。
- 证据：`outputs/maintenance/action_fov_scheme_a_20260929/{focused.log,focused.exit,integration.log,integration.exit}`；`before/` 保存修改前代码／测试／文档。未启动AE或WAM训练，没有GPU训练验收。

以下0.1–0.5保留相机轴手形版本 `fixed_camera_delta_latent_v1` 的历史过程与真实失败结果，不能视为当前腕局部版本的验收；第1–8节保留9月28日更早表示的运行记录。

### 0.1 2026-09-29 当前回归结果

**2026-09-29 Singer 回退核验**：按用户要求撤销 Claude review 后实施的改动，保留 review 前 C=4、57D fixed-camera 表示、C=4预算和已确认的loss系数1。原 smoke／训练验证／恢复三个脚本与修改前备份逐字节一致；新增 launcher、preflight、diagnostics、native verification 及对应测试已撤销，官方共享启动器等4个文件恢复到原Git版本。路径及clip档位恢复原设置（129／65／33），不启用257／273或历史覆盖强制检查。恢复后的配置／数据／packer／runtime／checkpoint合同CPU联测 **166 passed（58.70s）**；未运行GPU或训练。证据：`outputs/maintenance/singer_rollback_verification_20260929/{inventory.json,regression.log}`；回退前文件和patch保存在 `outputs/maintenance/undo_claude_review_20260929T175027/`。Claude补充review原文保留，相关方案仍待用户决定。本地残留的Singer实施说明通过备份后同步远端修正，不据旧测试宣称新方案就绪。

- 新几何＋归一化 28 项通过；旧推理兼容 43 项通过；新 fixed runtime 16 项通过；扩展 streaming／inference／model 回归 135 项通过。
- 数据 agent 的独立集合 117 项通过，数据 chain 已接通；真实 state/future 57D Piecewise-Asinh 统计尚未生成。
- 上述集合覆盖范围不同且可能重叠，**不得直接相加为总通过数**。它们证明当前接线和回归范围，不证明 AE 质量、真实统计或正式训练已就绪。
- 最终主合并联测为 **147 passed（65.75s）**，覆盖 geometry、normalization、data、codec、runtime、config、contract 和 legacy configs。该 147 项与此前扩展 streaming／inference／model 的 135 项是另一集合，可能重叠，**不能相加**。新配置名为 `rbs_wam_ar_v0_2_fixed_camera_delta_latent_v1`。

代码清理复审（2026-09-29）：已停用的 V0.5／V0.1 配方及旧归档 YAML 不在当前工作树中；官方 Cosmos3 配置保留。重复 `_c`／`_multitask` TOML 合并为旧表示回归用 `ar_v0_2.toml`，新增明确选择当前表示的 `ar_v0_2_fixed_camera.toml`。历史注册名与真实运行快照保持原语义。修复数据审计仍默认读取已删除 V0.5 配方、入口漏传 K=8 两处残留；V0.1 推理兼容入口要求显式提供历史 TOML。配置／runtime／合同回归 34 项通过，新增布局／官方 TOML 解析及旧推理检查 14 项通过，两集合有重叠，不相加。未启动 GPU 或正式训练。

预算复审：模型固定 C=4 后，packer 仍沿用 C=1 最坏预算，导致可容纳的两个长 clip 被拆开。现已统一为 `joint_chunk_cond_v1_two_pass_full_us_c4_v2`，保持双 pass、noisy 条件行、512 padding、60K 和最多 4 clip。640×368、129 帧、100 文本 token 的两样本预算为 40376。数据／布局／恢复联测 138 项及审计 4 项通过；旧预算版本、缺失版本和旧 lookahead 均拒绝恢复，避免批次边界改变却声称精确续训。更大 pack 的 GPU 显存和多节点恢复尚未复测，不能宣称已通过 OOM 验收。

### 0.2 新 AE 首轮有界试验：未通过

输出目录：`outputs/fixed_camera_codec_smoke_20260929_v2/`。首轮使用 64 个 train episode、训练 2000 步，在 24 个 held-out episode 上评估：

| 指标 | 右手 | 左手 | 诊断门槛 | 结论 |
|---|---:|---:|---:|---|
| 重建 mean | 7.15 mm | 9.43 mm | ≤5 mm | 未通过 |
| 重建 P95 | 16.06 mm | 21.77 mm | ≤15 mm | 未通过 |
| 32 次换轴重编码 | 61.52 mm | 41.99 mm | ≤10 mm | 未通过 |

三项门槛要求左右手分别同时满足；本次所有指标至少有一侧超限，尤其重复换轴漂移明显。因此不能生成正式 codec 合同，不能据此拟合最终 normalizer，也不能启动 WAM 正式训练。这里的 5mm mean、15mm P95 和 10mm rebase 是本轮项目工程准入诊断值，不是 flow matching 的数学要求，也不是任何官方标准。后续两组追加试验已经完成，结果见第 0.4 节；本轮小规模试验已结束，结论仍为未通过。

### 0.3 固定轴几何长链自审：修复后通过

FP32 长链自审最初在第 40 块触发旋转 `det(R)` 校验失败。原因是连续积分与换块中的有限精度误差累积后，旋转矩阵逐渐偏离 SO(3)；该失败不是 action 数量、坐标轴定义或阈值设置导致。

修复后，相机／双腕在每次旋转增量积分和换块变换后都统一经过既有 rotation6D helper 重建正交旋转；平移仍按固定块首相机轴直接相加，`ΔR=R_t R_(t-1)^T` 与左乘语义不变。没有放宽原来的有限值、正交性、行列式或参考误差阈值。

新增 100 块／3200 帧 FP32 长链与 FP64 参考对照，逐步检查旋转有限值、正交性和 `det(R)=1`。修复后 geometry＋主 runtime 40 项通过。该 40 项与第 0.1 节的几何／runtime 集合存在重叠，不能与 28、16、135 或其他集合直接相加。此结果只关闭刚体长链数值稳定性问题；新 AE、真实两套 57D normalizer 和正式训练准入仍保持未完成状态。

### 0.4 新 AE 两组追加试验：均未通过，停止小规模调参

| 试验 | 输出目录 | 右／左重建 mean | 右／左 32 次 rebase | 结论 |
|---|---|---:|---:|---|
| fit10000 | `outputs/fixed_camera_codec_fit10000_20260929/` | 3.54／4.86 mm | 82.47／117.47 mm | 重建改善，rebase 失败 |
| consistency1_10000 | `outputs/fixed_camera_codec_consistency1_10000_20260929/` | 3.95／5.13 mm | 47.69／69.45 mm | 左手重建与 rebase 均失败 |

统一比较保存在 `outputs/fixed_camera_codec_consistency_compare_20260929/comparison.json`。identity 路径在不改变坐标轴时也随重复 encode/decode 漂移，因此当前证据指向 codec 的重复 E/D 不稳定，而不是第 2.0 节固定轴换块公式错误。这里测的是 static-shape stress，用于放大 codec 一致性问题；它不是实测 WAM rollout，不能把数值直接解释为真实控制 rollout 漂移。

32 次递归 E/D／rebase 压力测试对应 `generated` 模式：本块预测末态经过换轴和重编码后继续作为下一块自生成 S。`gt` 与 `pred_history` 每块都从真实边界观测重新构造 S，不会沿用上一块的重编码 state；其中 `pred_history` 只把历史 V/A 换成预测，当前 U/S 仍为 GT。因此不能把 generated 压力结果外推成所有模式具有同等漂移，也不能据此判定 57D fixed-camera 表示的数学定义不可行。

两组追加试验结束后仍未达到本轮工程准入诊断值，当前停止小规模 AE 调参并保留质量问题。全模式尚未验收，新 AE 未定版，不能生成正式 codec 合同；真实 state/future 57D Piecewise-Asinh normalizer 仍未产出，未启动正式 WAM 训练，也不得写成 `training ready`。

### 0.5 现有 hand 指标口径限制

当前 `ar_v02_evaluation` 与 overlay 的 GT 手来自 GT action latent 经 AE 解码后的关键点，并非数据集原始关键点。因而现有 hand MPJPE 不包含 GT codec 相对原始关键点的重建误差，不能用来替代上述独立 AE 重建验收。主输出标记 `hand_metric_target='decoded_gt_action_latents'`、`hand_metrics_include_gt_codec_reconstruction_error=False`，overlay 标记 `gt_hand_source='decoded_gt_action_latents'`。这些字段只明确已有指标与可视化的目标来源，不表示新增评测方案或已有 AE 质量通过。

## 1. 2026-09-28 历史正式训练（旧动作表示）

当时正式配置为 `cosmos3_joint_video_hand_pose/configs/ar_v0_2_multitask.toml`，官方 Nano 初始化，CP1/FSDP8、梯度累积 1。它使用旧块首 state／旧 hand AE、C=1…4 和 `L_video + 0.7 L_action`；二次尺度校准与历史加噪关闭。推理为双路独立 schedule 的常规 30 步。以下记录只描述该次历史运行。

| 参数 | 该次历史运行值 |
|---|---|
| 更新步数／保存间隔 | 1200／600 |
| warmup／cosine 周期 | 100／1200 |
| packing | 60,000 token、每卡最多 4 clip、512 padding 余量 |
| 窗口与时间 | H=15 历史 chunk；历史训练 C=1…4；K=8；当前块不计入历史 |
| 内存回收 | `every_n=1, warm_up=0, gc_level=2`，从第一步完整回收 |
| 正式输出 | `outputs/joint_video_hand_pose/ar_v0_2/formal_multitask20h_60k_20260928T140938/` |

正式运行从官方权重重新开始，已于 **2026-09-28 19:36:04 完成 1200 次更新**。第 600、1200 步 checkpoint 均保存成功，日志明确输出 `Done with training.`；这不代表模型质量评测已完成，也不代表 `fixed_camera_delta_latent_v1` 已训练。最终 checkpoint 在上述正式输出目录下 `training/joint_video_hand_pose/ar_v0_2/multitask20h_60k_20260928T140938/checkpoints/iter_000001200/`，包含 model、optim、scheduler、trainer、dataloader。

**W&B 记录缺失及补传撤销（2026-09-28）**：本次训练误设 disabled，原始 run ID 为 `b159gyif`，原训练目录没有 `.wandb` 数据库。视频/action 分项历史没有持久化，无法恢复；不是有完整本地 DB 尚未同步。

训练结束曾从 train.log 补传 164 个 rank0 总 loss 点、114 个速度点和 1200 个日志梯度点，并通过 API 核验。但这不是完整训练指标，用户要求删除，该在线补传 run `5swvvyak` 已删除，API 确认不存在。当前没有本次有效在线链接，也没有可用完整 DB 替换。总 loss 最后记录为 0.0958，不据此宣称收敛。

正式输出目录 `diagnostics/wandb_backfill/` 保留解析指标、上传及删除审计；其中 `.wandb` 是事后补传新生成的数据库，不是原训练历史，不能再次上传后称为完整记录。权重、原始 train.log 和数据均保留。后续正式多任务配方默认已改为 online；本次 recipe_snapshot.toml 与 config.yaml 保留 disabled 的真实记录。后续仍需落实独立本地分项指标备份，不能仅依赖在线上传。

## 2. 多任务数据与质量

原始数据：`/mnt/lzh/egoVerse/datasets/full/human/{mecka,aria,scale}/`。历史全库报告约 61,240 episodes／1151 小时，本轮没有重扫全库。报告在 `/mnt/lzh/egoVerse/datasets/quality/EGOVERSE_DATA_QUALITY_REPORT.md`。

本轮从 Mecka 100h 候选清单（4295 episodes、705 个原始任务标签）筛选，使用一致的 640×360、30fps、双手字段。Aria 的时间／caption 问题及 Scale 的几何差异未在本轮统一，因此未混入。

筛选要求：双手关键点和腕部有效率至少 0.999，骨长异常率不超过 0.01，平均关键点画内率至少 0.7，腕部速度 P99 不超过 2m/s。4261 episodes 通过，34 个因速度被排除；只保留至少 4 个 episode 的任务后为 4258 episodes／704 标签。seed=42，按任务轮转选取；基于标签首词的分组不是语义任务分类。

| split | episodes | segments | 原始任务标签 | episode 小时 |
|---|---:|---:|---:|---:|
| train | 744 | 4409 | 143 | 17.33694 |
| heldout | 119 | 642 | 77 | 2.73052 |
| 合计 | 863 | 5051 | 143（去重） | 20.06746 |

保留片段时长约 19.58416 小时。按用户要求仅 train／heldout；原候选 83 个 heldout 与 36 个 test 合并为验证集，保留 `source_split` 可追溯。train 与 heldout 的 `(user, scene)` 组合隔离，不代表用户、场景或任务各自完全隔离。713 个少于 65 源帧的短片段被排除；863 个数据路径已核实。文本只取 segment caption，避免未审核的 episode 总描述。

质量证据与边界（数据清单仍可供新方案重处理；以下 normalizer／codec 数值只属于旧表示）：

- 初始人工抽查 8 个 episode 的 24 帧，覆盖做饭、园艺、编织、包装、焊接、雕刻、清洗等；投影大体一致，指尖仍有不确定性，不代表全库标注精度已验证。
- 有效窗口审计覆盖全部 4409／642 段，均保留有效窗口；块首 state 规则额外排除数为 0。
- **旧** state normalizer 仅用训练集 1,685,704 条边界样本拟合，只有 18 个腕部维度，相机槽为 0；FP32 往返最大误差 4.17e-7。它不是新 57D state normalizer。
- **旧** future pose normalizer 使用 1,822,790 条原生相邻帧增量，保持 B3 平移 zero/std、旋转 q01/q99；往返最大误差 5.96e-8。它没有新 `Δz` 语义，不能复用。
- **旧** hand codec 每段抽 3 帧，train 左／右平均重建误差 1.97／2.08mm，heldout 2.00／2.15mm；其输入去除了腕旋转，与新 q 不同。这些数值不构成新 AE 质量证据。
- **旧** loader 每个 task/split 读取 1 个完整 clip，共 220 个（143＋77），视频解码、旧 57D action 有限值与 `2×(T−1)` 未来 action 行数检查通过；未逐帧重解码全库，也未验证新 fixed-camera action。

必需产物位于 `outputs/data_expansion_20260928/`：候选 `episodes.csv`、`segments.csv`、`tasks.csv`、`summary.json`；选择和审计脚本也保留于此。候选 summary 不代替后续训练验收结论。

历史 `prepared_v02/` 内含冻结 `valid_windows.json`、`eval_windows.json`、旧 `chunk_state_normalizer.json`、旧 `future_frame_delta_normalizer.json`，以及 data/codec/loader 审计、拟合样本和报告。数据清单与历史证据保留，不能清理；其中 normalizer 和 codec 不能用于 `fixed_camera_delta_latent_v1`。新方案必须输出独立路径和新 hash，禁止覆写此目录后改写历史。

## 3. 有效 epoch 与训练预算

动态 packing 的 global batch 不是固定 8 或 32。有效 segment 暴露次数定义为：

`所有 rank 累计消费的 segment 窗口数 / 4409`

新数据 60K＋完整 GC 的 50 步成功检查共消费 797 个 clip，平均每步 15.94：

| 更新步数 | 预计窗口访问数 | 平均 segment 暴露次数 |
|---|---:|---:|
| 600 | 9564 | 2.17 |
| 1200 | 19128 | 4.34 |

每次访问随机抽窗口，因此不是完整视频遍历次数，也不是独立时间覆盖率。正式运行应以实际消费计数重新计算。旧 148 段小数据的 182.43 epoch 外推已经不适用。

600 步保存作为首个效果检查点，1200 为当前目标；需看验证集物理误差及视频—action 一致性，不能仅凭训练 loss 判断。50 步检查去首步的最慢 rank 平均约 16.56 秒／更新，不含正式保存和评测，仅作预算参考。

## 4. OOM 原因与容量选择

发生过两个不同的内存生命周期问题，不能概括为 packer 塞入过多 clip。

| 事件 | 原因 | 修复与证据 |
|---|---|---|
| 初始 70K 检查，完成 5 步后第 6 步 OOM | backward 重计算时显存不足；训练 clean KV 的闭包／bound method 引用环使跨步对象滞留 | `_build_tf_memory_state` 改用 weakref／WeakMethod，不 detach 梯度；预算随后比较 55K／60K |
| 新多任务 60K，完成 30 步后第 31 步 OOM | 继承的 ManualGarbageCollection 约第 20 步关闭自动 GC，浅层 gc_level=1 未回收老代引用环 | V0.2 从首步每步 gc_level=2；旧代对象回收回归通过，随后 50＋1 全程稳定 |

第二次失败在第 20 步后，rank 1 步末显存从约 18.53GiB（第 21 步）持续涨至 45.72GiB（第 30 步）；不能用此前 12 步 smoke 宣称长期稳定。修复未改变 loss、加噪或 KV 梯度，也未修改框架全局 GC 默认值。

55K／60K 历史容量比较：8 卡各 12 步，覆盖 C=1…4，目录 `outputs/joint_video_hand_pose/ar_v0_2/packing_sweep_20260928T130645/`。

| token 上限 | 最大实际 token | allocated 峰值 GiB | 设备采样峰值 GiB | 平均秒／更新 |
|---|---:|---:|---:|---:|
| 55K | 48056 | 51.29 | 60.83 | 14.19 |
| 60K | 55918 | 54.53 | 64.49 | 14.99 |

60K 提高容量但不等于吞吐更高；该短测的 action 吞吐约 175.81/s，55K 约 186.18/s。按用户希望利用显存选择 60K，并以修复后的新数据 50＋1 作最终稳定性依据。按最坏 C=1 双完整 pass 计费、加 512 余量、最多 4 clip；两个长 129 帧 clip 的约 63K 组合不会准入。

## 5. 验收证据与修复索引

### 5.1 历史训练启动检查

根目录：`outputs/joint_video_hand_pose/ar_v0_2/final_acceptance60k_20260928T132800/`。

| 检查 | 结果与产物 |
|---|---|
| 固定实际 VAE latent，两次独立 2 步，8 rank | `fixed/summary.json`：`exact_gradient_digests=true` |
| 新多任务 50 次更新→保存→退出→独立恢复→第 51 次更新 | `multitask_resume50_fullgc/`：SUCCESS；模型、优化器、scheduler、trainer 与 8 卡 dataloader 状态恢复 |
| 数据进度 | 8 rank trace 连续覆盖 iteration 0…50 |
| 数值与内存 | C=1…4，loss／原始梯度有限；allocated 峰值 54.57GiB，步末约 14.95GiB，无逐步累积 |

失败的 `multitask_resume50/` 与早期 `reliability_50plus1_20260928T040716/` 日志保留用于定位；它们不代表修复后的最终状态。正式启动脚本检查验收成功、源码指纹、数据审计和 GPU 空闲后才启动。

### 5.2 布局、缓存与回归

- 已修复多样本接线：恰好两个 clip 被误判为官方 transfer、文本 offsets 跨样本、padding timestep 范围和 pack 试放副作用。
- 已修复训练语义：逐块 `[U,V]` VAE 编码、块首相机系 state、跨 rank 全局有效样本平均、mask 后再计算平方以避免无效数据 NaN、embedding 优化器覆盖及 checkpoint 合同。
- 已修复恢复与推理：按保存窗口确定性重建、空文本 prefill、角色索引、条件 timestep、缓存容量、generated 边界解码重编码与 state 坐标变换；评测区分块内误差与累积漂移。
- 紧凑 noisy 流仅有未来 V/A query；clean KV 仍可求梯度。`domain_aware_linear` bias 的 FP32 汇总及统一 masked key tile 顺序解决 bf16 路径差异，未以放宽阈值处理。
- 完整 Nano、640×368、GT 历史、shift5 下，首块及 C=1…4 的 k=16／17 首次淘汰边界和相应残块，共 4540 项严格比较零差异，采样均为 30 步。边界测试从给定 15 个 GT 历史块开始，不是完整自由生成长 rollout。索引：`outputs/joint_video_hand_pose/ar_v0_2/review_fixes_summary_20260928.json`。
- 流式接口 smoke：`streaming_aligned_20260928T104643/summary.json`（位于同一 ar_v0_2 输出根目录），C=1…4 共 8 块含残块；使用预编码 GT latent，不代表在线 RGB 端到端性能。
- 代码格式整理时 47 文件 AST 一致，330 项非 GPU 回归通过，日志 `outputs/maintenance/cleanup_20260928/`。这是当时快照，不冒充最终源码重新跑完整 GPU 矩阵；之后生命周期专项另有 `tests/test_ar_v02_gc_policy.py`。不同批次重叠测试数不相加。

`generated` 的 `local_*` 表示转换到 GT 块首相机系，仍含累计误差；必须和 GT state 重置的块内指标分开标注。

### 5.3 尚未完成与暂缓

| 项目 | 边界 |
|---|---|
| 正式训练后的质量评测 | 未完成，不能宣称 joint/state 已改善效果；没有独立消融就不归因 |
| 完整 Nano 缓存扩展矩阵 | 全历史模式、双 schedule、全部残块组合尚未全跑；4540 项不代替完整矩阵 |
| 完整 Nano 离线 CLI 全链路评测 | CPU 指标／MP4 和接口 smoke 不等于完整 GPU CLI 评测通过 |
| 旧实验诊断 | 旧实验已按用户要求清理；不再安排其补充实验，新版质量评测另行指定 |
| 推理速度与 48 块端到端延迟 | 用户明确暂缓，不承诺实时或已通过延迟门槛 |
| state／joint／校准／历史加噪独立训练消融 | 暂缓，出现具体问题再做 |
| 梯度累积大于 1 | 当前正式配方为 1，不声称其他配置已验收 |

## 6. 文档归并与追溯

原 code_map 已并入设计第 7 节；数据扩展、多任务接入、训练预算、OOM、packing 容量与 review 已合入本文。原始 Markdown 快照保留在 `outputs/maintenance/docs_v02_before_merge_*.tar.gz`，不再作为当前状态入口。实验日志、数据、模型、运行目录及本地审计图表未删除。

## 7. 官方复用与自定义实现全链路审计（2026-09-28）

审计对象为远端 `e395cd1` 加当前未提交改动；以仓内同步提交 `849d3ca` 为上游差异基线。范围包括 45 个项目／验证 Python 文件的入口、继承和覆盖清单，20 个上游改动文件，以及正式运行的最终配置、启动命令、日志和 checkpoint 目录。主审与三个只读专项审计分别检查启动、监控／恢复、数据／loss、推理／上游补丁。不是对全部第三方代码逐行证明正确；没有重训或执行完整 Nano GPU 验收。

### 7.1 哪些沿用官方，哪些是项目定制

| 环节 | 实际调用链与判定 |
|---|---|
| 正式训练 | `src/train.py:16` 通过 runpy 调用 `cosmos_framework.scripts.train`；官方 TOML 解析、Trainer.train、optimizer/scheduler、DCP 仍在使用。原生 CLI、DCP、wandb_log、wandb_util 与同步基线逐字节一致，没有重写正式训练引擎 |
| 启动包装 | 本次使用 outputs 下的临时 `start_formal.py` 组织门槛和 torchrun；它不是训练循环，但没有检查 online／指标持久化，工程交付不合格 |
| 动态 packing | `JointChunkPackingDataLoader → RecoverablePackingDataLoader → PackingDataLoader → JointDataLoader`，仍用官方装包循环，扩展预算与恢复 buffer |
| 主干与噪声 | 项目模型继承 OmniMoTCausalModel；复用官方噪声分布，再映射到 chunk。VAE、文本 tokenizer、基础投影和优化器仍为原生实现 |
| U/S/V/A、joint、cache | 扩展官方 PackedSequenceBuilder、MemoryState、denoise/replay 接口。官方 target-only/no-text 第二遍不支持当前 action＋多样本组合，不能把项目 compact 一概视为不必要重写 |
| 数据与 loss | EgoVerse 57D 编解码、有效窗口、逐块条件、整体 action loss／全局样本权重属于任务适配。未发现当前 CP1、累积1配置下全局均值重复缩放 |
| checkpoint 恢复 | 复用 DCP；项目补充 dataset/worker RNG、待消费 pack buffer、归一化绑定等，基线缺这些接口，扩展有依据 |
| 日志 | 确有过度替换：重写 W&B callback 行为、全局过滤 wandb.log，并缺少独立本地指标持久化 |

### 7.2 已确认问题与处理方向

以下是审计发现，除已注明的正式多任务 online 开关外，**不表示修复已经完成**。行号均相对远端仓库当前快照。

| 优先级 | 发现、证据与影响 | 修复方向 |
|---|---|---|
| P1 | **正式记录没有验收门槛。** 原始 recipe_snapshot.toml 明确 disabled；`src/config.py:33` 替换 basic logger 组，`src/wandb_metrics.py:336` 只发 W&B。临时 `outputs/joint_video_hand_pose/ar_v0_2/final_acceptance60k_20260928T132800/start_formal.py:5` 起检查数值/数据，但不检查日志落盘。造成整次分项历史丢失 | 正式路径保留官方 logger 生命周期；项目 callback 增量记录三个 loss；独立本地备份；在正式入口跑短验收并检查真实 history。多任务 TOML 已改 online，其他部分仍待修 |
| P1 | **全局 monkeypatch 静默丢弃官方指标。** `src/wandb_metrics.py:117` 替换整个 wandb.log；白名单丢掉官方 `timer/iter_speed`、`timer/tokens_per_second`、`mem/*`、sample_counter 等。即使开启 online 也会发生 | 删除全局过滤，仅控制项目自己写出的字段；需要少图表应配置 dashboard 或明确关闭特定高开销 callback |
| P1 | **流式 RGB 未走官方归一化。** `src/ar_v02_streaming.py:359` 接受 uint8 RGB，转 dtype 后直接 encode；官方 `omni_mot_model.py:4563` 单视角入口会先将整数图像归一化。CPU 复核像素 `[0,128,255]` 原样进入 VAE，而预期约 `[-1,0.00392,1]` | 复用官方单视角编码入口或其归一化函数，明确 dtype/range 合同，并增加真实 RGB 入口测试；预编码 latent smoke 不覆盖此问题 |
| P2 | **覆盖 logger step-end 顺带删掉计时清理。** `src/wandb_metrics.py:146` 空实现；官方 `utils/callback.py:520` 原本执行 training_timer.reset。源码中无其他调用；CPU 模拟 1200 次后仍保留 1200 个计时值 | 恢复原生生命周期，不将 logger 清理职责随过滤一起删除；这是主机计时列表增长，不是此前 GPU OOM 的已证实原因 |
| P2 | **诊断遗漏新 embedding，且 logger 修改梯度。** `src/wandb_metrics.py:196` 分组漏 action_state_embed/vision_condition_embed，却称 all_selected；`:298` 会清洗梯度。当前 contract 先检查坏梯度、原生 force_finite=True，未证明当前更新被改变；关闭保护并设置 force_finite=False 时 logger 仍会覆盖意图 | 诊断覆盖真实 optimizer 参数并只读梯度；清洗/裁剪交回原生 GradClip |
| P2 | **上游长度补丁过宽。** `joint_dataloader.py:1485` 将任何 IterableDataset 长度都变成0。有限6样本、batch2也从3变0。当前无限 action stream 可以特殊处理，但不能影响所有有限流 | 显式识别无限流；有限数据沿用官方长度 |
| P2 | **同一 token cap 在两层有不同准入口径。** `ar_dataset.py:123,301` 未加 reserve，`ar_v02_dataloader.py:105` 加512。129帧、368×640、文本100时预算31500；cap31700会被dataset接纳、被packer按32012丢弃 | 共享统一预算函数，必要时退短tier，统计拒绝原因；未证实当前60K真实数据已触发 |
| P2 | **流式非零 source_start 与训练窗口 RoPE 不一致。** `ar_v02_streaming.py:276,373` 将绝对边界直接写位置，而训练使用窗口内起点。相同内容 source_start=64 使首U时间偏移25.6 | 分离绝对元数据与模型窗口原点，或显式限制支持范围；默认0不触发，实际GPU输出影响尚未测量 |
| P2 | **旧入口仍会启动错误配方。** 主项目 README:8 指向 ar_v0_2.toml/data_v2；该配方和 ar_v0_2_c.toml 仍 disabled、小数据。scripts/README 也保留已过时的暂停说明 | 当前正式入口统一到 multitask 配方；旧配方明确回归用途，不能混作正式默认 |
| P2 | **验收用手工循环与正式生命周期不完全相同。** `src/smoke_train.py:434` 自行循环调用原生 training_step；`scripts/validate_ar_v02_training.py:170` 用该入口并强制 disabled。可以验证局部数值，却不能证明正式入口的日志/保存顺序完整 | 数值夹具保留；正式启动验收另走同一个官方 Trainer.train，通过 callback 采集检查，不再把smoke等同正式链路 |
| P2 | **源码冻结声明超过实际覆盖。** `scripts/validate_ar_v02_training.py:45` 的指纹漏 joint_dataloader、omni_mot_model、grad_clip、DCP 和验证脚本本身，临时启动仍称源码无变化 | 根据实际依赖形成可恢复的源码快照和manifest；临时脚本收敛为版本化入口，不能只用不完整hash作为同版本证明 |

### 7.3 配置透明性、条件风险与已知边界

- **实际学习率**：最终配置 base=2e-5，但继承 `src/config.py:245` 的倍率。共享／视频组峰值为 **8e-5（4×）**，action及新增embedding为 **1e-4（5×）**，再按warmup/cosine变化。它来自旧实验继承，不能只报告base lr；没有证据将倍率本身判为算法bug。
- **验证**：最终 `run_validation=False, dataloader_val=None`，本次没有周期性 heldout validation。这与此前“训后评测”的安排可兼容，但不能声称已在600步做效果验收。
- **梯度累积**：当前只支持1；大于1的全局计数需要生产接入，现状会显式报错。文档已声明这个限制，非当前静默梯度错误。
- **恢复完整性**：原生 strict resume 仍允许缺失某 rank 的 dataloader/RNG 状态时继续，项目需要更严格检查；当前1200步目录存在8个dataloader文件及model/optim/scheduler/trainer元数据，未运行完整恢复，不能称已证明完整无损。
- **计数／回放**：自定义 grad_clip 累计触发次数未持久化，resume后重置；旧 monitoring.render 读取normalizer路径但不校验prepare保存的hash，文件替换会造成回放来源不一致。这些不等于当前模型参数丢失。
- **codec 身份**：当前checkpoint合同绑定normalizer/窗口，但未绑定手形codec内容。没有确认当前heldout泄漏；codec训练来源和权重hash仍需纳入可复现合同。
- **不误报**：当前logging_iter=1，LR/梯度/loss步数一致；原生也没有已证实的默认本地JSONL兜底（原生备份是save_s3条件分支）。因此不能把缺失历史解释成“删除了官方本地数据库”。DomainAwareLinear的FP32 bias累加属于有理由的数值补丁，但全局兼容性仍需专项验证。

### 7.4 本轮验证与下一步顺序

远端 `outputs/maintenance/native_reuse_audit_20260928/` 保存inventory、logger过滤复现、计时清理复现和CPU测试日志。配置／W&B兼容／loss／训练合同的 **38 项现有CPU测试通过**（116.92s）；它们未覆盖真实正式入口的指标持久化，测试通过不能推翻上述发现。流式RGB、source_start和有限IterableDataset问题由CPU方法级检查确认，未运行真实VAE/Nano性能测量。

修复顺序：先收回全局日志覆盖、恢复官方生命周期并补正式入口日志验收；然后修流式RGB预处理和上游补丁作用域；最后统一配置／预算／身份绑定并补恢复与边界回归。保留本次1200步权重与运行快照，不为修日志重训，不把合理任务适配整体推倒重写。此轮为审计，训练和推理实现尚未按本表修改。


### 7.5 审计后的修复与清理（2026-09-28）

本节更新上述审计快照的处理状态。用户要求先修复，下一次实验由用户另行指定；本轮没有启动训练。

- **官方能力复用**：正式入口仍仅注册配置后转交官方 CLI，继续使用官方 Trainer、optimizer/scheduler/DCP。恢复官方 WandBCallback、wandb_2x、计时、参数统计及 packing 指标；删除全局 wandb.log 过滤和空 step-end 覆盖。项目 callback 只增量记录任务指标，梯度诊断只读，包含新 embedding。
- **分项 loss 持久化**：rank0 每个完成步追加 `loss_metrics.jsonl`，保存真实 total/video/action、step 及可用诊断，flush/fsync，恢复不覆盖。非有限 loss 显式标记，不填造有效值。本地备份不代替正式在线记录验收，也不能补回上一轮未保存的分项历史。
- **数据与推理**：dataset/transform/packer 共用含 512 reserve 的预算；RGB 复用官方归一化，浮点输入按已归一化合同；流式绝对源帧元数据与相对 RoPE 分离；有限 IterableDataset 保留原生长度。
- **恢复**：项目 callback 补充逐 rank dataloader/RNG 完整性检查，缺失即失败；加载前拒绝不支持的梯度累积。现有 1200 步 checkpoint 的 8-rank 元数据预检通过，这不等于已重跑完整 GPU resume。
- **配置**：从官方 Nano 直接构建 V0.2，删除旧 overfit → AR V0.1 继承。三个 V0.2 配置名统一多任务数据、online、600 步保存。待定新配方的 LR 倍率明确为空（各组 1×），不再隐含继承旧 4×/5×；这没有更改已完成实验的快照，下一轮实际 LR/步数仍待用户确认。base state_normalizer 仍被公共数据接口使用，保留它不代表继承旧实验。
- **验证脚本**：源码指纹覆盖项目源码、官方 framework 和验证脚本；不再限制为单一节点，保留 GPU 空闲检查。手工数值 smoke 仍只是项目夹具，不代替官方 Trainer.train 的完整生命周期验收。禁止再用 outputs 中旧临时 launcher 启动正式实验。
- **小项**：裁剪触发计数改名为 `trigger_count_since_process_start`，明确恢复后从新进程累计；旧公共回放工具核验准备时的 normalizer hash。
- **清理**：旧 overfit、AR V0.1 配方/启动脚本/文档、旧结果及速度基准目录已删除；本地旧视频同步删除。删除清单在 `outputs/maintenance/native_reuse_fix_20260928/deleted_pre_v02.json`。当前 V0.2、原始数据、codec、共享预训练权重保留。旧推理/投影 CLI 停用，被当前代码或数值参考使用的公共函数继续保留。

尚未宣称完成：真实 GPU 上官方正式生命周期的在线 W&B 指标验收；当前修复后完整 Nano 缓存/评测扩展矩阵；codec 新身份字段的兼容迁移（当前旧 checkpoint 缺该字段，未强行改变读取合同）。这些不通过重启旧实验解决，待下一次授权实验前按需验收。推理速度继续暂缓。

整合 CPU 回归：**283 passed，6 skipped，61 warnings，176.01 秒，exit=0**。覆盖日志生命周期、配置、loss、预算、恢复合同、流式输入/位置、有限迭代器、投影辅助和推理数值。6 个 skip 为原生不支持的测试参数组合，未冒充通过。日志：`outputs/maintenance/native_reuse_fix_20260928/cpu_regression.log`。官方训练 CLI、DCP、wandb_log、wandb_util 与同步基线 `849d3ca` 对照仍未改写。

## 8. 官方入口 GPU 验收（2026-09-28，进行中）

用户授权全部剩余测试，不启动大规模训练。训练验收直接调用项目配置注册薄入口，再转交官方 CLI / Trainer；测试专用保存间隔不改变正式配方的 600 步间隔。

### 8.1 真实训练、日志和保存恢复

- Tdebug4：`outputs/joint_video_hand_pose/ar_v0_2/native_acceptance_20260928T123111Z`，4 步保存后恢复至 6 步，两个进程均 exit=0。官方 DCP 实际恢复 model、optim、scheduler、trainer、dataloader，保存成功。
- 在线验收 run（测试run `sushybps`，2026-09-30已删除）：API 已核实第 1–6 步 video/action/total 与本地 `loss_metrics.jsonl` 数值一致，三项都是真实训练记录，不是补传；run 已 finished。
- Tdebug2：`native_reference_20260928T123111Z` 连续 6 步参照，exit=0。参照 run（测试run `4eh22e94`，2026-09-30已删除） 的第 1–6 步三项 loss 也已通过 API 与本地逐项核实。
- **发现恢复一致性缺陷**：8 个 rank 的数据索引、sample_id 和 packed_global_id 全部一致，但恢复后 sigma 统计不同。不能把进程成功退出当作精确恢复通过。
- 原因：torchdata 恢复 worker iterator 时仍临时抽取一次 CPU `int64.random_()` base seed；DCP 已恢复的主进程 RNG 因此被推进。官方 video/action sampler 在 CPU 抽样，后续噪声错位。项目原来仅保护 buffer 重建的 RNG，没有覆盖 super 初始化。
- 修复：只在恢复路径保护整个 iterator 初始化的 Python/NumPy/CPU Torch RNG，保留官方加载器。`tests/test_ar_v02_resume_parent_rng.py` 真实双 worker、预取 2、snapshot 间隔 1/3 回归 **2 passed（20.14s）**，旧路径稳定复现额外抽样，修复后 RNG 与连续读取一致，后续 8 个样本的随机输出也一致。
- GPU 复验：`native_resume_fixed_20260928T123111Z` 从原第 4 步完整 checkpoint 继续第 5、6 步，exit=0。8-rank 数据 trace 和全部已记录 sigma 统计与连续参照完全一致；三项 loss 最大相对差异 0.0464%，通过预设 rtol=0.005 / atol=1e-6 断言。bf16 结果不声明逐位一致。第 6 步完整 checkpoint、8 个 dataloader 文件及旧日志前缀保留均已检查。修复后在线 run（测试run `jo2w72oj`，2026-09-30已删除） 第 5–6 步三项 loss 已与本地核实。对照结果为该目录的 `resume_comparison.json`。首次外部恢复尝试被合同检查拦截：配方仍带官方初始化的 embedding skip；测试明确改为 `keys_to_skip_loading=[]` 后通过，不放宽检查。旧失败证据保留，不覆写原 run。

### 8.2 推理与 codec

- **完整 Nano cache／重算矩阵 42/42 通过**：结果根目录为 `outputs/joint_video_hand_pose/ar_v0_2/acceptance_inference_20260928T203700/`，总报告 `final_acceptance.json`，逐 case 路径和逐项断言见该报告与各目录 `case.json`、`comparisons.jsonl`。使用正式 iter1200 的原始嵌套 checkpoint、36 层完整 Nano、bf16、真实空间尺寸；C=1…4、全部有效尾块、gt／pred_history／generated、shift5／independent 两种 schedule，固定 30 步、H=15。共 792 个 chunk、269,280 项比较，最大绝对差与最大相对 L2 均为 0，无数值失败；覆盖窗口未满、k16、首次淘汰 k17、之后及尾块，逐步 flow/Euler、逐层 prefill/refresh/保留 KV、noisy 不写缓存和独立 rollout。原阈值未修改。此结果完成第 5.3 节原列为待完成的该范围 GPU 缓存与 CLI 验证，不代表速度、全验证集效果或训练消融通过。
- **参考语义与独立路径**：按 `design.md` 第 6.1 节，reference 从序列开头保留完整前缀至当前块末尾，以 H=15 mask 重算；“截断”只去掉未来，**不是仅取最近 15 块重新编码**。后者会丢失保留深层 KV 形成时吸收的更早上下文，不是本次等价性结论。当前设计已明确此区别，但未单列最初“截断参考”措辞的需求变更记录；此处明确验收口径，不能把本结果外推到最近 15 块重编码算法。
- **动态 path witness 通过**：`path_witness_v2/summary.json`、`path_witness_v2.exit=0`。缓存路径执行 33 次实际模型前向（含 text），使用持久 `JointKVCache`；63 次参考 API 调用实际执行 63 次 clean 与 63 次 noisy 前向，使用 `TeacherForcingMemoryState`。参考期间禁止访问持久缓存，且两路初始化和实际输出 storage 不同。完整矩阵两路分别更新预测与 generated 下一块条件，并非同一输出自比。零差异与相同权重、联合 mask、绝对位置及一致的有效 KV 遍历顺序相容；两路仍共享模型和 mask 定义，不能据此独立证明共享定义正确。初版 witness 错把参考 API 次数当全部模型前向次数，计数断言失败；原证据保留，修正观察器计数后重跑通过，未改模型或数值门槛。
- **有限时长与续验记录**：两项 C4/tail3/independent（gt、generated）原进程在 18 个完整块已通过后，于第 19 块触及 3600s 上限，exit=124。未终止仍运行任务；确认退出后，从两路分别保存的预测恢复并回放前缀重建缓存，只采样缺失的第 19 块。两项续验均 exit=0，与原尾块已有的 106／110 项比较记录精确一致。详见 `recovery_c4_tail3_{gt,generated}_independent/summary.json`。因此 42 项中有两项是分段完成，不声称全部单进程不间断完成。队列中 `FileExistsError` 是已委派目录的所有权跳过，非数值失败；以最终逐 case 结果聚合，未重复启动占用中的 case。
- **真实入口及视频产物**：`rgb_stream.summary.json` 的 16 项真实 RGB 流式入口测试通过，覆盖 uint8／归一化 float、C=1…4、source_start=0/64。官方离线 CLI 在两个不同 heldout episode 上完成 6 组采样／评测／overlay（含 C3 尾块和非零起点），产物位于 `offline/`、`offline_second/`；`mp4_validation.json` 核实六个 MP4 均完整解码为 65 帧、30fps、1280×368。长缓存矩阵使用重复的真实编码块，不能替代长 RGB rollout 质量验收；视频视觉结论见 8.3。
- **运行期间源码 drift 已逐项核实**：初始 41 文件 hash 对照仅有两处主线协调修改：① `src/dataloader_state.py::_initialize_child_iterators_once` 在恢复 iterator 初始化周围保护 Python／NumPy／Torch CPU RNG；② `src/model.py::_compute_whole_losses` 返回值仅新增 `train_objective_numerator=logged_loss.detach()` 与 `train_objective_denominator=torch.ones_like(logged_loss.detach())` 及说明注释。两者不参与推理数学路径；cache、模型推理、codec 和数值阈值无此期间改动。完整前后 hash 和 patch 见 `source_drift_verified.json`、`dataloader_state.drift.patch`、`model.drift.patch` 及最终报告。补充 runtime mtime 扫描也只发现这两文件；mtime 扫描不冒充未预存文件的历史 hash。验证脚本在启动前按当前数据配置修正过 eval manifest 路径，未改共享 runtime。
- Codec sidecar：`scripts/verify_ar_v02_codec_sidecar.py`，17 项 CPU 测试及实际 iter1200 检查通过，证据在 `outputs/maintenance/native_acceptance_20260928/`。当前内容、架构和旧严格合同一致；旧 checkpoint 未存 codec 权重 hash，历史身份无法补证，sidecar 也尚未自动接入 runtime。
- 官方支持多节点 HSDP，节点软件版本已核对。Tdebug4 GPU0 与 Tdebug6 GPU6 的双节点 NCCL 通信测试通过：1/8 MiB 各 10 轮，两端 exit=0，实际使用 NET/Socket；结果在 `outputs/maintenance/native_acceptance_20260928/nccl_2node_20260928T131827Z/summary.json`。这只证明通信正确性，不代表通信性能或 16 卡训练已验收，也不把 8-rank 恢复结果外推为 16-rank 精确续训。

恢复相关整合 CPU 回归为 **50 passed（80.85s）**。后续进一步核对发现：项目 loss 已按全局有效样本数缩放，官方 logger 的默认回退分支却再按每卡 pack 样本数加权，导致 `train/loss_avg` 与项目 `loss/total` 不同；训练梯度和项目三项 loss 不受该日志问题影响。修复使用官方既有 `train_objective_numerator/denominator` 接口，提供 detached rank contribution 和分母 1，不改写 callback。相关 **61 项 CPU 测试通过（62.41s）**，含真实双进程 Gloo、不等 pack 样本数、梯度不变性。

最后一次 8 卡在线短程复验 `native_logging_final_20260928T123111Z` **exit=0**，在线记录（测试run `oideyasa`，2026-09-30已删除） 已 finished。API 核实第 5–6 步三项 loss 与 JSONL 一致；原生 `train/loss`、`train/loss_avg` 与 `loss/total` 通过 rtol=1e-6 / atol=1e-8 断言，最大绝对差约 5.96e-8。第 6 步完整 checkpoint 保存成功；8-rank 数据 trace、sigma 与连续参照一致，三项 loss 最大相对差 0.0214%，再次通过恢复断言。证据为该目录的 `online_verification.json`、`native_logger_verification.json`、`resume_comparison.json`。Tdebug4 的一次额外 API 查询遇到 W&B service busy，切换 Tdebug2 只读核验成功；训练上传未失败。

最终只读检查：工作区及暂存区 `git diff --check` 通过；官方 CLI、DCP、wandb_log、wandb_util、WandBCallback 文件与 `849d3ca` 逐字节一致。Trainer 保留已有的恢复前 dataloader 绑定扩展，未声称整个上游目录零改动。


### 8.3 视频输出人工抽查

六份 MP4 均解码验证为 65 帧、30fps、1280×368；额外抽查中间帧，左右画面和投影均能显示，非零起点时间标签一致。生成历史下仍可见明显的预测手姿/轨迹偏差：这是现有权重的质量表现，本次管线和数值验收不等于模型效果通过。两个 episode 的短片也不代表全验证集评测。

本地仅提供视频链接，不嵌入截图：clip1 GT 历史（测试视频已清理）、clip1 生成历史（测试视频已清理）、clip2 GT 历史（测试视频已清理）、clip2 生成历史（测试视频已清理）。远端原文件位于上述推理验收目录的 `offline/` 与 `offline_second/`；本地镜像不是远端实验默认结果目录。
