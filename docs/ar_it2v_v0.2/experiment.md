# 完整segment接入与75k packing验证

2026-10-03。前半部分记录数据接入、动态packing、末块和日志的短测阶段；用户随后批准正式训练，启动信息见末节。旧V0.1准确标注为“最长97帧的随机短窗口训练”，用户要求停止时实际step2511，最后完整checkpoint为iter_000002500，权重和输出保留。W&B API核实旧run状态killed：[旧实验](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/7oimekos)。

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

## 新正式配方

用户已选4个空闲节点、HSDP8×4、CP1；max_iter1500，save500，warmup100/cycle1500；lr峰值1e-4/终点3e-5、weight_decay0.01。新注册`rbs_wam_ar_it2v_v0_2_ego100h_full_segments`，配置`cosmos3_ar_it2v/configs/ego100h_full_segments.toml`。从官方Nano开始，不恢复旧短窗口数据状态；首帧+文本、纯视频、GEN参数、C4/local16、原sigma配方和seed42不变。

最终TOML SHA256：`be92f1f816cdbf11839b539c49a40345e9b027a47c170d370c2d84e11dd60c49`。正式启动时重新检查四个节点资源，W&B online并经API核实真实step/loss/LR；短测交付时尚无新正式run；后续正式启动见末节。75k短测采用旧低LR优化器，不等于新1e-4配方的长期稳定性或质量验收。

官方LambdaCosine理论lr：step0=0、100=1e-4、750=6.89187566636e-5、1500=3e-5。W&B继续原生`optim/lr`并额外记录实际update前lr范围；旧run API已存在2511个optim/lr点，问题是曲线不易找到，而非调度器未运行。官方AdamW已实现weight decay，官方Nano SFT配方本身也使用wd0；旧值为0不代表功能缺失，新.01按用户选择。

RoPE继续官方绝对latent时间位置、文本偏移及fps modulation，30fps步距24/30=.8；没有每块重置。CMD Stage1同样使用绝对current_start，但其Predict2.5关闭fps modulation，不能把另一基座的频率直接覆盖Cosmos3。只迁移Stage1，未混入少步/长视频蒸馏；位置及KV实测见上。

## 2026-10-03 独立复核

以`e385d41`为审查起点，分别核对数据/packing、模型数学、训练生命周期与推理缓存，并对照官方实际实现。未发现完整segment、文本配对、token预算、DF逐块timestep、packed隔离、loss全局segment等权及绝对RoPE路径的新增阻断问题；75k短测回执已与真实日志/退出回执核对。

发现并修复两项P2：恢复训练时原先重置首次10步loss基线，现从已有monitor记录恢复截至checkpoint的有效loss历史，缺失明确报错，新进程显存窗口重新计数；长视频推理原先无限保存KV、读取后才裁窗口，现复用官方有限DualKVCache，C4/local16使用4槽，保留相同12 latent可见历史和绝对坐标，不随rollout长度累积全部缓存。

新增7项CPU用例，最终与launch相同PYTHONPATH、CUDA关闭，全量`tests/`为87 passed/121 warnings，24.60秒；原80项均通过。本次未重复GPU试验、未启动正式训练。AGENTS已明确复杂组件先查官方能力，只有实际缺口才写最小适配。

审查边界：尾latent的r/4是覆盖率加权，不能精确拆掉VAE内部混入的补帧成分；已有GPU gate为8卡、0 workers、3步低LR，未实测32 rank×3 workers和1e-4长期稳定性。75k mask修复前后的首步梯度范数不同，但两次未开启deterministic_mode、未捕获sigma/epsilon/RNG，不能据此认定backward错误，也不能声称只是舍入差异。正式配方和启动确认要求不变。

### 提交记录

- `6468e76`：AGENTS明确复杂组件优先复用官方能力。
- `9e66bcb`：恢复时保持原始loss停止基线及有效历史窗口。
- `a058689`：长视频推理改用官方有界KV缓存，保持原可见性。

- `4fbb1e9`：停止旧run、准确修正短窗口训练范围。
- `9bf20b9`：完整segment、真实末尾监督和有保护的官方动态packing。
- `3420afa`：批准的75k超长保留策略、4节点/1500步及用户选定优化器；75k短测代码源为完整hash `3420afa9194b3f4f4797f09bdcb658cba649ea57`。
- `dd32448`：复用官方metadata-run构造器，消除token² mask临时内存；v2代码源`dd32448cfe573463df7b582a8131b4ec6f69d6fb`，不改mask规则或训练配方。


## 2026-10-03 正式训练启动

用户在独立review和修复交付后明确批准启动。2026-10-03 11:25–11:26北京时间分别经MCP启动Tdebug2/3/4/5，各8张H800，rank0/1/2/3；HSDP8×4/CP1，NCCL Socket、eth0内网TCP，master10.3.12.18:29873。启动前各节点GPU空闲、无compute进程，CPU quota120核、affinity200核，空闲内存约1.4TiB以上。复用官方启动器和Trainer，四端有独立claim/启动/退出记录；不得重复启动或自动恢复。

本次名为`ar_it2v_v0_2_full_segments_75k_lr1e4_4nodes`，代码源`11c04ee7049b63b8fb1a3d348fbf419df9f208ff`。从官方Nano初始化，未恢复旧短窗口run；1500步/save500、75008 token、max_samples=None、完整segment及批准的超长retention策略、4节点/32卡、每rank3workers、lr峰值1e-4→3e-5、warmup100/cycle1500、wd.01、seed42均已核实实际config.yaml。实际配置SHA256 `a535c763fa7b7ddbe9ea56bbf542fe44b7f2c857eef6b4be31795656674e16b0`；TOML hash仍为上述be92f1f…，manifest和retention记录hash沿用覆盖回执。

W&B online，[本次run 7aa7je4o](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/7aa7je4o)。API已核实state=running、真实step2/video_loss/实际LR上传；step2实际更新LR为1e-6，官方optim/lr记录调度后的下一步2e-6，两者时点不同。启动回执及API有限字段见[evidence/formal_launch.json](evidence/formal_launch.json)。

| step | 全局segment数 | token填充率 | 训练区间秒 | video loss | 裁剪前梯度范数 | 实际更新LR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 210 | 97.35% | 77.71 | 0.336398 | 1.28339 | 0 |
| 2 | 133 | 97.04% | 37.49 | 0.333856 | 1.35049 | 1e-6 |
| 3 | 115 | 96.97% | 36.28 | 0.328138 | 1.31909 | 2e-6 |
| 4 | 147 | 97.66% | 38.70 | 0.320381 | 1.37467 | 3e-6 |

前4步峰值allocated62.46GiB/reserved74.08GiB，loss/梯度均有限，无STOPPED或节点退出。API step1到2时间差36.86秒。排除首步初始化，初期36–39秒/步；含保存与数据波动暂估1500步16–20小时，第500步约5–6小时。此为启动阶段估计，不以4步loss下降宣称收敛，不保证后续所有长pack均不会OOM；既有停止规则照常执行。

输出：`/mnt/lzh/cosmos-ar-it2v/outputs/pretrain/20261003T032429Z/rbs_wam_ar_it2v/ar_it2v_v0_2/ar_it2v_v0_2_full_segments_75k_lr1e4_4nodes`。启动计划/四端日志/API回执：`outputs/maintenance/ar_it2v_v0_2_full_segments_preflight_20261003T032429Z/`。


## Step1000 完整 segment 视频预览

使用本次完整segment正式run的 `iter_000001000/model`，train/test各3条完整动作段（train420/186/390帧，test722/309/373帧），原始30fps、对应整段文本、seed42、35步去噪、CFG1、历史刷新噪声0.02；长短是不同完整segment，无窗口截取，仅去除VAE对齐补帧。
Tdebug6空闲GPU0–5并行，6/6正常退出，18个MP4已核对帧数与帧率；每段41.53–151.53秒，峰值allocated32.22–34.37GiB。预览实现提交1749434，不修改训练源或配方，不计算质量指标。
产物及selection/完整checkpoint校验/media_validation回执位于 `outputs/visualization/ar_it2v_v0.2/step1000/full_segments/`；服务器回环18768通过原连接转发，GT/生成双栏，完整文本与实际时长可见，页面不发布样本身份或训练文件。
