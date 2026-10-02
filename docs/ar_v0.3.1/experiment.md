# AR V0.3.1 单因素对照

本次正式实验由用户于 2026-10-02 批准。

## 配方

- 仅将块 k<L 的视频与 action loss 分子置零；分母、有效样本归约均沿用 V0.3.0，目标块梯度尺度等于原 S 分量。保留后续块通过历史注意力回传到前缀的梯度。
- 从官方 Cosmos3-Nano DCP 起点开始；数据、顺序、seed=42、全局 packing batch、逐块 σ 与低噪声前缀抽样、57D/PCA15/C4/K8/H15、腕权3、clip档位沿用 V0.3.0。
- 全模型 lr=1e-4；官方 LambdaCosine warmup100/cycle3000/f_min0.3；max_iter3000，每500步保存；HSDP8×2、CP1、NCCL内网TCP。
- 额外记录前缀/目标 × 视频/action 的未加权坐标 MSE（仅日志，不参与优化）；原8个字段日志保留完整原始统计口径。
- 500/1000/2000/3000各做 σ_small=0.02/0.05/0.1 的 heldout8 gt/generated 与 train16 gt，72jobs/96NPZ，同步数V0.3.0同表比较腕误差、整体光流方向余弦、幅值比、PSNR和第17块漂移。光流沿用320×180整体dot/norm口径。
- 500/1000评测后向用户报告；明显更差时由用户决定是否提前停止。数值/OOM/显存增长的原停止规则保持。

## 启动

- 名称：formal_prefix_numerator_only_t273_20261002T061300Z
- 输出：/mnt/lzh/cosmos-EgoWAM/outputs/joint_video_hand_pose/ar_v0_3_1/formal_prefix_numerator_only_t273_20261002T061300Z
- Tdebug1 rank0 / Tdebug3 rank1，各8张H800；内网10.3.12.57:29861，eth0 Socket。
- 启动回执：/mnt/lzh/cosmos-EgoWAM/outputs/maintenance/ar_v031_formal_preflight_20261002T061300Z
- 评测计划：/mnt/lzh/cosmos-EgoWAM/outputs/maintenance/ar_v031_formal_eval_20261002T061300Z；500/1000/2000/3000各72jobs、96NPZ，仅使用空闲Tdebug5/6。
- W&B：online，[hkog66xl](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/hkog66xl)。启动后由空闲Tdebug5上的W&B API核实：state=running，iteration/_step=2，video=0.218733、action=0.219669、total=0.438402，四项diagnostic MSE和8个原始字段均在线。
- 实际启动：2026-10-02 14:17北京时间；训练源89c79dd8910ae31b75cf1e72d434946a0e8d96de（实现1ab05f3及下述日志空值修复），实际config SHA256 af888957674e40dccaecc31298d0a59e5a4537570a528600c27564f23e94185f。
- 首批运行回执：本地已完成step4，单步17.42s（启动阶段，非稳定吞吐结论），全rank峰值allocated35.67GiB/reserved45.16GiB，梯度有限，STOPPED.json不存在、两端无退出回执。首步包含初始化开销76.01s；不以初始总loss跨配方判断质量。
- 四项raw-MSE日志保存在ar_v031_group_mse.jsonl，W&B API step2值：视频前缀0.932942/目标0.331257，action前缀1.114268/目标0.264290；各自按未加权有效坐标计数归约，与训练的固定分母目标不同。
- 后续监控每3小时一次，500/1000先报告同步数质量对比，由用户决定明显劣化时是否提前停止；否则继续3000及2000/3000评测。
- 参考：V0.3.0 formal_prefix_uniform_t273_20261001T182754Z，训练源9eac1859d2a94c6a4c41d12dee1f2ea92957bbdf，实际config SHA256 e7c5ab0150300247f139405470c16cbd668a8f22a8ce0cecf97795bda952ae66。

## 诊断动机与清理

step1000诊断4个真实global batch、共204个clip、每batch4次ε，无参数更新：前缀ε方向一致性0.903；期望梯度P/S范数0.13848/0.04534，约3.05:1；两组近正交、略有冲突。本次直接检验去掉前缀监督的影响，不以总loss回到0.13为验收依据。

上一轮过量临时准备、重复CPU/GPU测试套件、失败attempt与重复回执已按用户要求删除；只保留诊断最终报告与正式训练/评测产物。本次只做一次必要的固定分母数值验证，不另开短训。

## 首次启动的日志问题

2026-10-02 14:03北京时间，首次启动 formal_prefix_numerator_only_t273_20261002T054743Z（源1ab05f3）。16卡NCCL Socket通信、官方Nano加载及数据初始化成功；14:08首个step计算loss时，新增raw-MSE日志索引可选的 action_valid_mask=None，触发TypeError，两端退出码1。尚未执行backward或optimizer更新，没有checkpoint或训练loss上传，没有触发OOM或数值停止规则。W&B API确认 [4czrgihm](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/4czrgihm) 为failed，仅有运行时间，不作为实验结果。

修复仅给日志分支的可选mask加空值处理，None沿用原生全有效语义；优化loss、分母、采样均未改。一个CPU定点检查确认None、[None]、全True mask的四组raw-MSE统计逐位一致；未另开GPU短训。上述正式失败输出与两端回执完整保留。修复后使用新名称与独立目录，从同一官方Nano起点开始，保持相同seed和数据顺序。

## step1500 临时视频质量检查（2026-10-02）

固定 Pick-and-Place 训练3窗、heldout3窗，V0.3.0/V0.3.1均使用真实 step1500；σ_small=.02、seed42、联合30步、273模型帧/完整17块。冻结清单SHA256为 `f3c7bef5df5d22b63db120159e25c8d88b6aed5c1bd6567fe247076a9163374a`。这是六窗临时检查，不替代 heldout8/train16 正式评测；未改变训练或派发完整扫描。

24NPZ/12jobs全部通过原生验收，24长短MP4已渲染。Tdebug5/Tdebug6各6张空闲GPU；两版含权重读取的wall为677/514秒，单history纯采样平均86.61/88.04秒，峰值显存35.54GiB。CPU指标6workers、每进程1线程，24份一次聚合25.74秒。训练源、config与W&B沿用上文两次正式run。

短动作统计只取冻结裁剪内完整块（train共11块、heldout共14块），光流沿用320×180逐块U到末帧、整体dot/norm归约；PSNR先合并SSE/count。数值为V0.3.0 → V0.3.1：

| 范围 | PSNR | 方向余弦 | 幅值比 |
| --- | --- | --- | --- |
| train / gt | 15.060 → 15.023 | .039 → .032 | .910 → .812 |
| heldout / gt | 15.141 → 15.174 | .043 → .153 | .804 → .959 |
| train / generated | 11.187 → 11.717 | −.102 → .049 | .301 → .499 |
| heldout / generated | 10.912 → 11.586 | −.038 → −.031 | .164 → .497 |

结论：局部改善，但尚无稳定的短段运动提升。连续生成动得更多、PSNR略升，heldout方向仍近0；第一块heldout PSNR反而13.365→12.511、余弦.143→−.047。训练集目视仍有手停留在初始姿态、拿放过程不准确的问题，不能认定去除前缀监督解决了视频瓶颈。

版本产物：`outputs/visualization/ar_v0.3.0/step1500/pick_place/` 与 `ar_v0.3.1/step1500/pick_place/`；统一服务器网页 [版本切换入口](http://127.0.0.1:18767/)，保持既有布局。完整指标与完成回执在 `outputs/visualization/ar_v03_vs_v031_step1500_pick_place_20261002T134314Z/metrics/comparison.json`、`inference_complete.json`；原路径符号链接仅兼容已有回执。
