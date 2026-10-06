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


## 20步长segment近满预算压力测试：训练完成，最终保存OOM（2026-10-06）

用户认为3步不足，要求更多步骤及较长segment组合。源提交 `0263670`，Tdebug3单节点8×H800、HSDP8×1/CP1、49920源预算，沿用优化实现、官方Nano起点/full AC、seed42、原5000步scheduler/warmup100/LR/WD。仅临时入口将max_iter设为20并从正式train dataset筛选867个完整segment；未改生产代码、正式manifest、噪声或loss，未启用CP2/offload。20步仅覆盖warmup，不能证明峰值LR下长期收敛。临时入口/选择清单/检查全在ignored `tmp/ar_it2v_v04/longpack/`，Astra只读核对未发现正式配方污染。

选择口径：单段source token在[49000,49920)的540条，以及[24000,24800)可成对打包的327条；仍由官方Packer实际打包。全为原train划分，完整segment无随机裁短，超长段沿用已批准的90%→50%均匀保留规则。selection SHA256 `d5f8176ce58624bd87bfa3e4d3653c75602e231bd1e28675c3124cb366115900`，其上游49920 retention记录hash不变。临时factory先执行原dataset的全量清单/retention校验，再筛原rows并将selection hash写入数据恢复契约；CPU实例化检查通过。

北京时间08:57:18启动；训练20步在09:32:51全部完成。断开本地连接期间远端任务持续运行，无重复启动。实际160个pack、204条不同segment，其中116个单长段pack、44个双段pack；原始时长13.13–55.03秒，实际输入394–825帧（包含既定均匀保留策略），最大实际H/P Transformer长度99423。

| 指标 | 结果 |
|---|---:|
| 完成训练迭代 | 20（首步LR=0） |
| 平均source token填充率 | 98.38% |
| 最低单rank填充率 | 96.27% |
| 第3–20步耗时均值 / 中位数 | 100.79 / 100.75秒 |
| 第3–20步耗时范围 | 99.25–102.52秒 |
| 第3–20步全局source token吞吐 | 3896.95 token/秒 |
| 训练阶段峰值已分配 / 预留 | 74.4626 / 77.6484 GiB |
| 第3–20步起始已分配范围 | 15.4580–15.4659 GiB |
| loss首步 / 末步 | 0.352114 / 0.148579 |
| 梯度范数范围 | 0.4877–9.2468，全部有限 |
| 末步LR | 1.9e-5 |

训练阶段没有OOM/STOPPED，也未见跨步已分配显存持续增长；不同长包的loss不要求逐步单调。相比之前普通混合样本3步，输入分布不同，不能将本轮100.79秒与95.79秒直接解释为优化回退。

**整个run未通过完整生命周期验收。** 官方Trainer在max_iter结束后仍自动保存最后一步（不受save500避免）；09:32:53 rank3在DCP `save_state_dict_worker → dcp.save → scatter_object_list` 的NCCL P2P分配处明确报告 `Cuda failure 2 'out of memory'`。09:33:42 launcher exit1，其他rank被launcher终止，GPU随后全部释放。`iter_000000020`目录存在但无有效shard/metadata/完成标记，不能当作可恢复checkpoint。全部失败输出保留，不自动重训/恢复。

W&B API核实 [huqeaezo](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/huqeaezo) 状态crashed、远端已上传到step19/loss0.171559/LR1.8e-5；step20以本地formal_monitor和训练完成日志为证，不能声称API确认了step20。完整证据及汇总在 `tmp/ar_it2v_v04/longpack_20step_20261006T005644Z/`。

后续优先排查/验证保存前释放allocator未使用缓存以及DCP首次NCCL通信额外分配；当前同步保存路径未调用empty_cache，仅加载后调用。缓存预留挤占NCCL余量是有代码/日志支持的候选解释，尚无保存时allocated/reserved快照，不当作已完全证实或已修复。正式四节点训练保持未启动；下一次验证需覆盖实际保存成功，不能仅以20步训练无OOM放行。


## 保存前缓存清理及20步长包复测：两次保存均成功（2026-10-06）

用户批准修复保存OOM后复测。源提交 `da0b20b623641cf8ab65132ea73e1cc12b58ebed`，仅V0.4增加官方保存前/后callback，保存前同步、GC并释放未用CUDA allocator缓存；不修改官方DCP/Trainer，不迁移活跃参数或优化器至CPU。6项针对性CPU测试通过，Astra/xhigh审查未发现阻塞项。新测试仍仅放ignored tmp。

Tdebug3单节点8×H800、HSDP8×1/CP1、49920预算、官方Nano新起点，使用上一节完全相同长segment选择清单、seed及配方；临时max_iter20/save10，正式5000步/save500不变。实际config TOML SHA256 `42cc6e280471680c0dfd6f1c1f707e2640de0bf83b2c9364289c630452967353`，selection SHA256沿用 `d5f8176ce58624bd87bfa3e4d3653c75602e231bd1e28675c3124cb366115900`。启动前8卡4–5MiB/0%，CPU quota120核、affinity200，未抢占其他任务。北京时间10:03:13启动，10:41:54 exit0，GPU全部释放。

8rank×20步共160条数据trace与失败长包run的有序sample IDs全部相同，无缺失/重复；仍是204条完整segment、116个单长段pack/44个双段pack，原时长13.13–55.03秒、实际输入394–825帧，98.3806%平均填充，最大Transformer99423 token。这里只确认样本顺序，不声称完整BF16噪声/梯度逐位一致。

| 指标 | 修复前长包run | 保存修复后长包run |
|---|---:|---:|
| 第3–20步训练耗时均值 | 100.787秒 | 101.138秒（+0.35%） |
| 同区间source token/秒 | 3896.95 | 3883.45 |
| 训练峰值已分配显存 | 74.4626GiB | 74.4626GiB |
| 训练峰值预留显存 | 77.6484GiB | 77.6465GiB |
| 训练迭代完成 | 20 | 20 |
| checkpoint/退出 | 最终保存OOM、exit1 | step10和20均成功、exit0 |

新run第3–20步中位数100.850秒、范围99.859–102.786秒；第11步即中途保存后的首步100.483秒，未见明显重分配停顿或跨步显存增长。训练timer在batch_end采样，**不含DCP保存耗时**；上表不是含保存的总吞吐。loss首末0.351959→0.149553，梯度范数0.3576–9.7736均有限，无STOPPED/OOM；仍只覆盖warmup，首步LR=0、末步1.9e-5，不是峰值LR收敛/画质结论。

| 保存点 | 各rank释放预留缓存 | 清理耗时范围 | 官方DCP写入耗时 |
|---|---:|---:|---:|
| step10 | 40.05–57.12GiB | 1.84–2.28秒 | 25.08秒 |
| step20 | 43.11–57.23GiB | 2.37–2.66秒 | 19.49秒 |

清理后step10各卡driver-free约60.38–60.46GiB；活跃allocated仍约17.1GiB，GC另外回收约6.6–7.0MiB不可达对象，并非allocated逐位不变。归还缓存为外部NCCL分配腾出空间，不等于训练激活节省40–57GiB，不支持恢复65K预算。两次保存8rank start/end回执完整。官方latest_checkpoint.txt最终指向iter_000000020，step10/20的model、optim、scheduler、trainer四组件metadata可由官方FileSystemReader读取，全部引用文件范围落在实际文件大小内，8rank dataloader状态齐全。scheduler未被引用的rank空shard是官方去重占位；未进行另一次恢复训练或全tensor校验。

W&B API核实 [93kgrebu](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/93kgrebu) 为finished、step20/loss0.149553/LR1.9e-5。完整计划、monitor、两次DCP、显存快照、API与汇总均在 `tmp/ar_it2v_v04/longpack_savefix20_20261006T020259Z/`；新开发脚本仍ignored。结论：本次单节点长包训练和实际同步保存生命周期通过，保存OOM在这次复测未复现；训练速度/峰值显存基本不变。四节点正式训练尚未启动，不将单节点20步结果扩展为四节点或长期训练保证。


## 正式训练：四节点32卡 / 5000步（2026-10-06启动）

用户批准清理准备阶段临时产物后启动正式run。此前本文smoke/容量短测均为准备记录，不计入正式训练；后续开发测试及详细回执只放ignored tmp，不再新增此类Git记录。本节为首次V0.4正式run，未从smoke或历史训练恢复。

- run：`parallel_tf_50k_4n_5000_20261006T031219Z`，北京时间2026-10-06 11:12:48–52分别启动四端。
- 源提交：`6a88c6a3d3a95fa76652b0a137339433504934cf`；生产入口`cosmos3_ar_it2v.train`、`configs/ego100h_parallel_tf.toml`，无longpack子集或临时配置。
- 资源：Tdebug1 rank0 / Tdebug2 rank1 / Tdebug3 rank2 / Tdebug4 rank3，各8×H800；HSDP8×4、CP1，NCCL Socket/eth0内网TCP，master `10.3.12.57:29914`。启动前四端GPU均空闲且无计算进程，CPU quota各120核。
- 配方：官方Nano新起点、seed42、49920源token动态Packer、完整train segment、C4/local32、全P监督/H仅间接梯度，历史LingBot公开代码噪声、P官方Nano waver/480shift5；full AC，无CP2/CPU offload。5000步/save500；warmup100、峰值1e-4→3e-5、WD0.01、GEN训练/UND冻结。
- 数据：32,010个保留train segment、原时间跨度89.5189小时，正常段全帧，超预算段按已批准retention策略；不使用短测867条筛选清单。Packer每GPU每步1个动态pack，32个全局pack，segment batch数随长度变化。
- TOML SHA256：`46709b11051480db4865734fe554ed30f8f7b4010b9ba86d3a6cc5e292585630`；实际展开config.yaml SHA256：`e1641d324e8315aaee5151e503e4bd8427d72bec96cf8a968a24eb79b08fb6c8`。
- retention records SHA256：`799a9cd94282d266328503144baadee61361e60dc97f912a1bda55a8a93f0eae`；episode manifest：`98d8abdff990cc3cf0b39a0d1ac0b0338b8d3be9e7b2ea7ed8757acfdacec41d`；segment manifest：`64655579936e40bf2774c0aca50993f1ef6b53c5fc37a44d70cb752950f9be60`。
- 正式输出：`outputs/train/rbs_wam_ar_it2v/ar_it2v_v0_4/parallel_tf_50k_4n_5000_20261006T031219Z`；计划及四端启动/退出回执：`outputs/maintenance/ar_it2v_v04_formal_preflight_20261006T031219Z`。

前两步已完成，step1/2分别处理129/109个segment，loss 0.423044/0.464697，preclip norm 7.3440/8.7298均有限。首步LR0、第二步LR1e-6，第二步训练耗时107.95秒、峰值allocated74.52GiB。无OOM/STOPPED；单个预热后step不是稳定耗时或收敛结论。停止规则沿用现有monitor；不自动重试或恢复。

W&B online实际run为 [j29qq2x1](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/j29qq2x1)。启动后API核实状态running，step1/loss0.423044/LR0已上传；上文step2与非零LR来自本次formal_monitor，不把本地进度冒充API进度。实际展开配置及learning_rate_receipt确认5000步周期、save500、49920预算及原生噪声设置均生效。
