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

- 名称：formal_prefix_numerator_only_t273_20261002T054743Z
- 输出：/mnt/lzh/cosmos-EgoWAM/outputs/joint_video_hand_pose/ar_v0_3_1/formal_prefix_numerator_only_t273_20261002T054743Z
- Tdebug1 rank0 / Tdebug3 rank1，各8张H800；内网10.3.12.57:29861，eth0 Socket。
- 启动回执：/mnt/lzh/cosmos-EgoWAM/outputs/maintenance/ar_v031_formal_preflight_20261002T054743Z
- W&B：online，alexlzh431564/joint_video_hand_pose；实际run链接待启动后API核实。
- 参考：V0.3.0 formal_prefix_uniform_t273_20261001T182754Z，训练源9eac1859d2a94c6a4c41d12dee1f2ea92957bbdf，实际config SHA256 e7c5ab0150300247f139405470c16cbd668a8f22a8ce0cecf97795bda952ae66。

## 诊断动机与清理

step1000诊断4个真实global batch、共204个clip、每batch4次ε，无参数更新：前缀ε方向一致性0.903；期望梯度P/S范数0.13848/0.04534，约3.05:1；两组近正交、略有冲突。本次直接检验去掉前缀监督的影响，不以总loss回到0.13为验收依据。

上一轮过量临时准备、重复CPU/GPU测试套件、失败attempt与重复回执已按用户要求删除；只保留诊断最终报告与正式训练/评测产物。本次只做一次必要的固定分母数值验证，不另开短训。

