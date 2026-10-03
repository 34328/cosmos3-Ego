# 完整segment接入与75k packing验证

2026-10-03。本轮只修改并验证数据接入、动态packing、末块和日志；新正式训练尚未启动。旧V0.1准确标注为“最长97帧的随机短窗口训练”，用户要求停止时实际step2511，最后完整checkpoint为iter_000002500，权重和输出保留。W&B API核实旧run状态killed：[旧实验](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/7oimekos)。

## 实现与数据覆盖

复用官方Cosmos3 Trainer、PackingDataLoader、caption tokenizer、优化器/调度器和DCP。适配负责Zarr、原segment边界及文本、VAE对齐/末块和可恢复数据状态；不随机裁短、不跨段，不修改C4/local16及sigma配方。预算75008（75k向上128对齐），max_samples_per_batch=None；原生严格成本上限不再静默丢段。

正常段使用全部原帧。仅单条超预算段按100%→90%→80%→70%→60%→50%检查，保留原首尾与时间跨度；50%仍超限才排除。实际source indices、effective_fps、真实帧数、对齐帧数和排除原因都有记录。原始fps为30，抽帧段使用实际effective_fps输入RoPE，不用30fps加速。

| 数据 | 原始segment数/小时 | 纳入数/小时 | 排除数/小时 | 保留原时长 |
|---|---:|---:|---:|---:|
| train | 32355 / 95.5101 | 32333 / 94.9597 | 22 / 0.5504 | 99.4238% |
| test | 1180 / 3.5689 | 1178 / 3.5078 | 2 / 0.0611 | 98.2887% |

训练集31266段保留100%帧，345/301/234/154/33段分别保留90/80/70/60/50%；测试集1133段保留100%。纳入最大成本train74935、test74928，均严格小于75008。统计覆盖33535段，16workers/3.91秒/8574段每秒，代表性串并一致；全清单dataset实际构造预检通过。逐段记录保留于`outputs/maintenance/full_segment_retention_75008_20261003/summary.jsonl`，摘要及manifest/tokenizer/records hash见[evidence](evidence/full_segment_coverage.json)。

VAE仅补0–3张末帧，整段连续编码；真实尾部latent按有效帧占比进入分子/分母，解码裁回真实长度，部分末C4块保留。VAE尾latent混合真实与补帧，不能逐RGB分离其内部pad影响，此权重不声称完全消除该影响。packed样本attention隔离、逐块timestep和数据恢复合同均有CPU覆盖。

## 验证

- 全量CPU：与launch相同PYTHONPATH，CUDA关闭；最终80 passed、121 warnings。由原39项增加41项，覆盖完整边界/文本、packing隔离、超预算策略、末块权重、数据恢复、旧配置复现与metadata mask等价。无新增失败。
- 真实VAE/末块/KV：Tdebug6 H800 GPU0，从官方Nano加载，只forward、不更新。96帧补到97，解码裁回96；187帧补到189，48个latent，末块3个latent。相同xt/sigma下整段与streaming相对RMSE分别1.481%/1.248%，最差块cos≥0.999836，满足预定bf16 RMSE<3%/cos>0.999；不是逐位相同。绝对RoPE坐标差最大0.0009765625，末块为0。峰值allocated32.72GiB/reserved35.21GiB。详见[真实回执](evidence/partial_kv_gate.json)。
- 75k官方Trainer短测：Tdebug5 H800×8，32个独立真实训练段（24段全帧、8段超长按70%保留），3步/save3，官方Nano起点；短测显式W&B disabled、lr2e-5/wd0，属于数据/资源验证。v1 step1填充98.28%、allocated71.47GiB/reserved75.83GiB，step2在稠密mask.sum时额外申请41.21GiB导致OOM；只完成lr0的1步，无非零LR更新、无checkpoint，不算通过。复用官方metadata-run构造器后，v2在全新目录`outputs/validation/full_segments/segment_packing_gate_75k_v2`验证，冻结清单/seed/顺序与v1相同，3步完成、exit0。

| v2实际step | global segment数 | 每rank数 | 平均token填充率 | 训练区间秒 | video loss | 裁剪前梯度范数 |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 16 | 2/2/2/2/2/2/2/2 | 98.28% | 70.69 | 0.36022 | 1.40575 |
| 2 | 52 | 9/8/3/10/8/8/3/3 | 94.25% | 60.91 | 0.34115 | 1.39389 |
| 3 | 8 | 1/1/1/1/1/1/1/1 | 93.82% | 34.73 | 0.29958 | 1.34167 |

峰值allocated62.07GiB/reserved71.70GiB，保留约7GiB硬件余量，维持75008预算，不继续提高；这是3步代表性布局验证，不保证所有未来pack形状和1500步永不OOM，仍执行原OOM/数值停止规则。各步batch随长度动态变化，不是固定sample batch。step3实际覆盖8个70%保留长段；3步共76次segment呈现（短测跨epoch，不是76个互异segment）。loss/梯度均有限，三个不同batch的loss下降不作为收敛/质量改善证明。训练区间时间不含初始权重加载、dataloader等待和checkpoint保存。

iter_000000003官方marker、model/optim/trainer/scheduler各DCP metadata及其引用分片均完整；8个rank的数据状态可读、global_id=3、预算75008和pending buffer保存。Ns不一致时保留官方全局segment等权公式8×Ns/Nall。v1/v2首batch样本/CFG trace与tokens一致，CPU mask逐位等价；GPU loss/梯度并非逐位一致，已原样记录，不能据此宣称GPU逐位等价。详见[完整回执](evidence/packing_75k_gate.json)。

初次65536启动误用系统torchrun，在import阶段退出，0步/0更新；已修正PATH为cosmos3环境，未重复旧预算。输出完整保留在`segment_packing_gate_v1`。KV脚本的两次接口/manifest路径错误也保留，修复仅涉及验证脚本，不作为训练结果。

## 新正式配方（待短测结果后确认启动）

用户已选4个空闲节点、HSDP8×4、CP1；max_iter1500，save500，warmup100/cycle1500；lr峰值1e-4/终点3e-5、weight_decay0.01。新注册`rbs_wam_ar_it2v_v0_2_ego100h_full_segments`，配置`cosmos3_ar_it2v/configs/ego100h_full_segments.toml`。从官方Nano开始，不恢复旧短窗口数据状态；首帧+文本、纯视频、GEN参数、C4/local16、原sigma配方和seed42不变。

最终TOML SHA256：`be92f1f816cdbf11839b539c49a40345e9b027a47c170d370c2d84e11dd60c49`。正式启动时重新检查四个节点资源，W&B online并经API核实真实step/loss/LR；当前没有新正式run。75k短测采用旧低LR优化器，不等于新1e-4配方的长期稳定性或质量验收。

官方LambdaCosine理论lr：step0=0、100=1e-4、750=6.89187566636e-5、1500=3e-5。W&B继续原生`optim/lr`并额外记录实际update前lr范围；旧run API已存在2511个optim/lr点，问题是曲线不易找到，而非调度器未运行。官方AdamW已实现weight decay，旧wd0是本项目覆盖值，新.01按用户选择。

RoPE继续官方绝对latent时间位置、文本偏移及fps modulation，30fps步距24/30=.8；没有每块重置。CMD Stage1同样使用绝对current_start，但其Predict2.5关闭fps modulation，不能把另一基座的频率直接覆盖Cosmos3。只迁移Stage1，未混入少步/长视频蒸馏；位置及KV实测见上。

## 提交

- `4fbb1e9`：停止旧run、准确修正短窗口训练范围。
- `9bf20b9`：完整segment、真实末尾监督和有保护的官方动态packing。
- `3420afa`：批准的75k超长保留策略、4节点/1500步及用户选定优化器；75k短测代码源为完整hash `3420afa9194b3f4f4797f09bdcb658cba649ea57`。
- `dd32448`：复用官方metadata-run构造器，消除token² mask临时内存；v2代码源`dd32448cfe573463df7b582a8131b4ec6f69d6fb`，不改mask规则或训练配方。
