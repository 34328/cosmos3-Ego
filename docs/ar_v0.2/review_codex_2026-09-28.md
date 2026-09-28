# V0.2 新方案实现与训练检查

日期：2026-09-28。远端：`/mnt/lzh/cosmos-EgoWAM`，分支 `ar-video-action`。本轮保留原工作区改动，未提交。四个子 agent 分别检查数据／动态打包、损失／恢复、推理／投影、注意力／网络测试，主控整合并执行完整 Nano 短检查。

## 结论

新布局 `joint_chunk_cond_v1` 已接入远端。每块使用块首图像 U 与一个块首相机系 state S；未来视频和全量 action 联合去噪。新版已经完成真实多卡反向更新，但尚不能宣称本版全部验收完毕，更不能据短检查判断模型效果或实时性。

## 已定位并修复的问题

| 问题 | 后果 | 本轮处理 |
|---|---|---|
| 两个视频项被官方 attention 当成 control/target transfer | 每卡恰好两个 clip 时走错注意力分支 | 显式 joint override 不再走形状猜测分支 |
| teacher-forcing 的文本缓存仍只有一段 offsets | 打包样本间可能通过文本传递信息 | 逐样本文本范围，与 GEN sample_id 一起隔离；检查输出和参数梯度 |
| 短视频收到 `[batch,max_frames]` 的补齐 timestep 行 | 混合长度 pack 报错，单样本测试发现不了 | 按真实帧数截取，仅允许额外尾部为零；非零尾部报错 |
| 按整段视频 VAE 编码再切块 | 视频 latent 的因果上下文与逐块部署不一致 | 先选 C，再对每块 `[U,V]` 原始帧独立编码 |
| 旧 F0 state 统计套用于块首相机系 | 条件数值含义、解码坐标系错配 | 新建 train-only `data_v2`；相机槽固定零，腕部 18D 重新拟合 |
| 旧窗口恢复先随机取样，再覆盖 metadata | RGB/action 可能来自新窗口，state/source 却来自旧窗口 | 按保存的源索引确定性重建；校验状态、索引、统计 hash，保留保存的 CFG/text |
| 多卡／多 microbatch 直接平均局部均值 | 装入样本少的卡或 microbatch 被赋予更大权重 | video/action 分别按全局有效样本数归约；当前正式配置 accumulation=1，未规划的多步累积明确拒绝 |
| masked loss 在屏蔽前对极大无效值平方 | 无效坐标也可能溢出并污染梯度 | 先屏蔽再平方；条件、padding、有效监督各自处理 |
| 新 embedding 可能未进入优化器／恢复时被跳过 | state 或图像类型参数不学习、续训重置 | 检查参数覆盖，兼容官方 OptimizersContainer；续训预检类型参数与统计绑定 |
| 纯文本预填产生 `GEN indexes=None` | 正式缓存路径在第一步就失败 | 使用显式空 GEN 索引；保留单独文本预填 |
| U 索引、条件加噪和缓存容量沿用旧布局 | 条件被误更新、视频/action 错位或缓存容量不足 | 按角色、chunk、源索引重新定义；U/S 干净且无 flow loss |
| generated 直接沿用旧视频 latent 边界 | 下一块条件不符合单帧观测定义 | 完整 `[U,V]` 解码，末帧再编码；末态转到下一块相机系 |
| 评测跨块继续积分，或投影混用相机坐标 | 将累计漂移混入主指标，骨架投影偏移 | gt/pred_history 逐块从 S 解码；generated 单独报告累计漂移；内参与图像变换同步 |
| 动态 packer 用有副作用的 fit 计数 | lookahead 行为可能多取数据或错误计数 | 使用原生打包循环样本上限，保留 token cap 与回填 |

## 已有证据

- 新版统一 CPU 回归：**272 passed**，另有 **43 项旧 AR／loss 兼容测试通过**。包括布局、数据、打包、模型接线、缓存、采样、损失和训练恢复接口。不是沿用旧 F0 版本结果。
- GPU attention／小型网络回归：**26 passed**，追加原生缓存 **4 passed**、bf16 反向 **8 passed**，合计 **38 项**，Tdebug4 GPU0。包括多样本前后向、跨样本隔离、未来条件与当前干净答案泄漏检查、纯文本 prefill。
- 完整官方 Nano，CP1/FSDP8，Tdebug5：**2 个 optimizer update 成功，45 个样本步**。每卡装入 2–4 个 clip，覆盖 C=1…4、T=33/65/129，loss 与裁剪前原始梯度有限；峰值 allocated 显存 **58.17 GiB**。
- 动态打包真实数据：129+129 为 63,160 个原始双 pass token；129+65+33+33 为 63,716。实际 attention 对齐容量分别为 63,488／64,000。准入为每 pack 额外预留 512 tokens，对应预算 63,672／64,228，均低于 70,000 cap；该预留适用 CP1、关闭 CUDA-graph padding 的当前配置。
- 新 state 统计：62,040 个边界样本、148 个训练片段；9 个 held-out 片段及 81 项固定评测窗口保持不变；新增 state 缺标规则额外排除 0。归一化往返最大误差 `1.79e-7`。

主控 CPU 日志：`outputs/joint_video_hand_pose/ar_v0_2/validation_joint_chunk_cond_cpu_final_20260928.log`。

完整模型短检查：`outputs/joint_video_hand_pose/ar_v0_2/joint_cond_wiring_retry2_20260928.log`；结构化结果在 `outputs/joint_video_hand_pose/ar/smoke/smoke_ar_v0_2_20260928_015739/smoke_result.json`。

前两次失败日志保留：第一次停在 optimizer wrapper 检查；第二次暴露混长 timestep 问题。修复后第三次通过，不隐藏失败记录。

真实恢复检查位于 `outputs/joint_video_hand_pose/ar_v0_2/resume_2plus1_20260927T180331Z/`：`stage1.json` 保存 iteration=2，`stage2.json` 以 loaded_iteration=2 开始并完成 iteration=3。恢复后 8 卡共处理 18 个样本，原始梯度有限；保存阶段与恢复阶段峰值分别为 62.28／51.79 GiB。检查已实际退出旧进程，非在同一进程内模拟加载；8 rank 共 69 个待处理缓存样本按保存窗口重建并校验 source/state/hash。未与不中断训练做逐位等价对照。

离线入口为 `python -m cosmos3_joint_video_hand_pose.src.ar_v02_eval {sample,evaluate,overlay} --help`。指标 JSON、双栏 MP4 编码及读回已经 CPU 测试；完整 Nano GPU CLI 采样尚未验证。

追加缓存对照使用真两层 Qwen/MoT、C=1…4，每种运行 18 块、每块 30 次人工扰动输入，比较 flow 与各层条件／refresh K/V；覆盖首次淘汰。它尚不是完整 Nano bf16 的独立 Euler 轨迹。bf16 反向补测比较 288 个梯度张量，最大相对 L2 0.2484%，norm 比 0.999303–1.000366，未放宽阈值。

## 仍不能算完成的项目

1. **可靠性验收**：真实 2 步保存—退出—恢复 1 步已经通过，但设计要求的 50 次更新仍未完成，不据此证明长时稳定。
2. **完整 Nano 缓存验收**：仍需同输入逐步及独立 rollout 对照，并按设计阈值失败退出；小模型正确不能替代完整权重 bf16 对照。
3. **延迟验收**：流式 `StreamingJointSampler.step` 已实现，有界工作区与缓存逐块复用；仍需真实 30 步、至少 48 块及首次淘汰的端到端计时，不能把接口实现当作性能验收。
4. **训练计算冗余已修复**：第二遍仅计算未来 V/A，读取保留梯度的 clean 文本/U/S K/V；多长度、多样本隔离和末尾不足 chunk 已有回归。packer 暂保留双完整 pass 的保守准入预算，未借此增大 batch。
5. **梯度累积**：当前正式配置 accumulation=1。多 microbatch 的全局窗口计数 helper 有测试，但官方训练循环尚未自动预规划，修改为 accumulation>1 会明确报错，不会静默使用错误权重。
6. **效果结论**：尚未正式训练、做 V0.1 视频—动作失败来源诊断或完整质量评测。保持一个完整 V0.2 的安排，不启动旧消融组。


## 额外数值观察与审计

两次旧短检查的首步 global gradient norm 分别约 4.014 和 8.195，但旧日志未冻结真实随机窗口、CFG 文本和噪声，不能据此认定同输入，也不能断言是 kernel bug。下方新增真实固定输入的完整 MoT／checkpoint／FSDP 对照。

smoke 入口增加逐 rank JSONL 摘要：真实窗口起点、源索引、CFG token、原始输入、clean/noisy 实际 pack 和 sigma 的 hash。正式训练逻辑不引入此全量输入审计开销。本轮已用该入口执行完整 GPU 固定输入对照。

## 追加修复与 Claude 复审交接

**执行边界**：按用户最新要求，完成代码和短 smoke 后交由 Claude 复审。不启动正式训练；50 次更新、长 rollout 延迟验收也暂不执行。追加三路子 agent 因额度限制均报错退出，后续修复与验证由主控完成，不计作独立审查通过。

### 追加代码

- `src/ar_v02_compact.py`、`ar_v02_model.py`：紧凑 noisy query，仅保留未来 V/A。文本与 U/S 通过第一遍的可微 K/V 参与条件学习；保留 `compact_noisy_training=False` 全 query 参考路径。
- `packages/cosmos3/cosmos_framework/model/generator/mot/domain_aware_linear.py`：domain bias 先以 FP32 gather，再转回输出 dtype。重复 domain ID 的反向累计由此避免在 bf16 中逐次舍入；此前仅插入零梯度行就能导致约 1.35% 的梯度差。前向值、输出类型和 checkpoint 参数名不变，补丁已记入 UPSTREAM。
- `src/ar_v02_streaming.py`、`ar_v02_cache.py`：逐块控制 API 与固定 16 槽缓存；条件预填、30 次联合去噪、clean refresh，工作区不按全 clip 分配。修复无编号 `cuda` 与实际 `cuda:0` 被误判为不同设备的问题。
- `src/smoke_train.py`、`scripts/validate_ar_v02_training.py`：固定原始样本、实际 VAE latent、RNG 与初始参数，跨独立进程比较原始梯度。回放仍执行 encode 以保留元数据，再替换冻结 latent；此模式仅用于 smoke。

### 新证据（不可与旧结果简单相加）

- 最新 CPU 套件 **328 passed**，日志 `outputs/joint_video_hand_pose/ar_v0_2/compact_validation/final_cpu.log`。
- 紧凑/完整 query 前后向与 domain bias GPU 专项 **10 passed**；另一次含 CUDA 设备别名回归的专项 **3 passed**（其中两个 bias 用例重叠），日志分别为 `compact_validation/final_gpu.log`、`device_regression_20260928T100110.log`，均位于上述输出目录。覆盖 C=1…4、原生两层 MoT bf16 与 FP32 master gradient，未放宽误差阈值。
- 完整 Nano FSDP8 固定输入 **2+2 更新成功**：`fixed_latent_biasfix_20260928T095110/summary.json` 状态 success，8 rank 两步所有梯度摘要完全相同。两步全局原始梯度 norm 分别为 **5.2261854474863、4.957857902225794**，最大逐参数 norm 相对差为 0；未保存训练 checkpoint。这是短诊断，不是正式训练或效果实验。
- 上述每个进程运行均为 45 个样本步，每卡 2–4 个样本，C=1…4。
- 完整 Nano **流式接线 smoke 成功**：`streaming_nano_smoke_20260928T100110/summary.json`，C=1…4 各两块，共 8 块；C>1 的第二块只有一个 latent 时间组，覆盖末尾不足 chunk。每块 30 次联合去噪、32 次前向（首块额外 1 次文本预填单列），action 分别为 8×实际时间组条，视频/action/解码刚体位姿均有限。输入为已编码 latent，历史为 `gt`；不包含在线 RGB 编码、generated 闭环或窗口首次淘汰，因此不能据其约 2–5.6 秒计时宣称机器人端到端延迟验收通过。
- 前一次 `fixed_compact_20260928T094011` 比较失败的日志保留：仅冻结原始 RGB/RNG 时，不同进程的 VAE latent 仍有数值差别，因此不能称为去噪器同输入对照。最新结果只证明冻结实际 latent 后训练链路可复现，不证明 VAE 跨进程逐位确定。

### 尚未通过的严格检查——不可标记全绿

`nano_cache_smoke_20260928T094444` 在 chunk 1 条件预填、layer 2 K 上触发数值断言：bf16，shape `[1,241,8,128]`，relative L2=0.0040159458，max abs=0.078125。L2 低于 1%，但逐元素 `atol=0.01, rtol=0.03` 未通过，整体判定仍为失败。尚未确认是计算形状导致的 bf16 累计差异还是语义问题，**未放宽阈值**；完整 C=1…4、首次淘汰及独立 Euler 轨迹仍不能算通过。

重点复审：紧凑路径的 clean K/V 梯度是否完整、FP32 bias 累计补丁对上游的影响、固定输入审计边界，以及 Nano KV 严格误差来源。短 smoke 的可运行性与上述数值验收分别报告；恢复训练前应先明确该失败的处理结论。

## Claude review 后续修复（2026-09-28）

本节更新前述 KV 失败状态。正式训练仍暂停，用户另外明确将速度优化和延迟验收留到后面；本轮不运行性能验收。

### 原因与修复

失败来自我们适配层没有统一缓存/重算的**数值计算布局**：

1. PyTorch FlexAttention 默认分开遍历全可见和部分可见的 tiles。缓存的空槽及紧凑 query 改变 tile 分类，在线 softmax 的 bf16 舍入顺序也随之改变。原始路径首层 Q/K/V 完全一致，但注意力输出已有差异，深层 K 的 relative L2 最大约 0.0896。不是简单放大逐元素阈值就能结案。
2. 单独统一 tile 遍历顺序后，36 层条件预填 K/V 全部逐位一致，但首个 noisy flow 仍失败：参考路径是 `[text,noisy,clean]`，旧缓存路径是 `[text,clean,noisy]`，移除 U/S query 时还改变了 key 的 tile 边界。
3. 最终统一升序 masked tiles、noisy/clean 流顺序及绝对 token 的 tile 边界。紧凑训练只压缩 query，在 key 中使用无效零占位；缓存按绝对 ID 映射有界工作区，过期块只丢弃，不重算前缀。loss、RoPE、可见范围、H=15 与 30 步均不变。

对照证据：`precision_bf16_20260928T102311` 保留原始失败；FP32 诊断将误差显著降低，但深层仍未全部满足 1e-5，未将其误写成通过。只保留可见 token 的独立数学对照 `precision_canonical_20260928T103226`，以及单一 tile 顺序对照 `precision_ordered_20260928T103447`，均让原条件预填 36 层 K/V 逐位一致，明确定位到注意力数值路径。

### 已完成验证

- `aligned_nano_smoke_20260928T104300`：完整 36 层 Nano，原 latent 分辨率 23×40（对应 640×368），C=4 一个完整块加一个残块、每块 30 步。**680 项比较全部通过，最大绝对差与相对 L2 均为 0**；包含逐步同输入 flow/Euler、独立轨迹和各层缓存。阈值未变。
- `aligned_gpu_20260928T104300.log`：**19 passed**，包括新 tile 顺序回归、紧凑/完整路径输出和梯度、原生多层缓存测试。
- `aligned_cpu_v2_20260928.log`：**63 passed**，包括缓存逐步参考、首次淘汰及超过两轮槽位复用、C=1…4 残块和四种历史模式。另 `review_fixes_cpu_20260928.log` 的 **43 passed** 覆盖评测及新指标说明。
- `fixed_ordered_20260928T104642/summary.json`：本轮最终数值布局下，完整 Nano/FSDP8 的两次独立 2 步短测成功，8 rank 梯度摘要逐位一致、errors=0。两步原始全局梯度 norm 分别为 4.67123309060674、4.024337338644002；没有保存训练 checkpoint，不是正式训练。
- `streaming_aligned_20260928T104643/summary.json`：修复后的完整 Nano 有界流式接口，C=1…4 共 8 块通过，包含残块和固定 30 步。此项是接口 smoke，不是延迟验收。

边界诊断使用 15 个给定 GT 历史块做 clean prefill，再预测 k=16、k=17（首次淘汰）及尾块；每个预测块仍完整运行 30 步。该方式验证真实原分辨率下的边界数值行为，不宣称从首块开始的全部历史模式矩阵已完成。

保留的非通过记录：单空间 patch 的辅助 Nano fixture 在首层投影已出现 bf16 差异（与原分辨率失败起点不同），不能用它替代原分辨率验收；相应 `aligned_boundary_c*_20260928T104553` 标为 failed。原分辨率初次边界检查 `aligned_full_boundary_c*_20260928T105107` 中，已完成比较均为 0 差异，但参考重算路径因临时 MemoryState 闭包循环引用耗尽显存，整体仍判失败。验证脚本已在完成一次参考计算后释放实例回调、收集废弃引用，保留返回的 K/V；`tests/test_ar_v02_nano_validation.py` 的弱引用/返回值回归通过。修复仅作用于验证脚本，未改推理窗口或放宽阈值。

### 最终冻结代码验证结果

实际分辨率 640×368（latent 23×40）、完整 Nano、GT 历史、shift5，最终 **4,540 项严格比较全部通过，最大绝对差和相对 L2 均为 0**；每个运行的生产源码指纹前后一致。汇总：`outputs/joint_video_hand_pose/ar_v0_2/review_fixes_summary_20260928.json`。

| 运行目录（均位于上述输出目录） | 覆盖 | 比较数 |
|---|---|---:|
| `cache_final_smoke_20260928T111416` | 冻结最终代码重跑首个完整块＋残块，30 步 | 680 |
| `aligned_full_boundary_gc_c1_20260928T105616` | C=1，首次淘汰前 k=16／首次淘汰 k=17，各 30 步 | 710 |
| `aligned_full_boundary_gc_c2_20260928T105616` | C=2，k=16／17 及 1 组 latent 的残块，各 30 步 | 1,050 |
| `aligned_boundary_final_c3_20260928T110131` | C=3，k=16／17 及 2 组 latent 的残块，各 30 步 | 1,050 |
| `aligned_boundary_final_c4_20260928T110131` | C=4，k=16／17 及 3 组 latent 的残块，各 30 步 | 1,050 |

参考脚本只复用不可变的 attention mask，不复用重算路径的 activation 或 K/V；每步仍独立重算前缀。C=3/4 的前一轮 GC 版本因反复构造相同 mask 耗时过长，由主控中止自己的诊断进程、保留 `interrupted.json`，随后在最终版本重跑通过。上述结果不是全部历史模式、两种 schedule、所有残块余数的完整 Nano 矩阵，也不是延迟或效果验收；`full_acceptance` 仍为 false。正式训练与 50 步可靠性测试按用户要求继续暂停。

### 其他收尾及清理（最终）

- 旧 `ar_v0_2_c_no_chunk_state.toml` 标明暂停、不可运行，不恢复旧消融。
- 评测新增 `local_metric_scope`、`boundary_state_reset_to_gt`、`local_metrics_include_history_drift`；保留旧 metric key，明确 generated 的坐标转换不消除历史漂移。新用例注入边界 state 误差，确认 generated 指标保留该误差、GT 重置模式不继承它。
- 已删除被成功冻结-latent 对照替代的失败运行 `fixed_compact_20260928T094011/fixed_inputs/` 中 16 个 `.pt` 输入快照，释放 **2.732 GiB**。文件尺寸与 SHA256 清单在 `outputs/joint_video_hand_pose/ar_v0_2/cleanup_20260928.json`；原日志/摘要、成功对照输入、统计文件、checkpoint、数据与共享编译缓存均保留。
- 清理项目内可重新生成的 pytest 缓存及测试/验证脚本 `__pycache__`；不清理其他任务可能复用的共享 Torch/Triton 编译缓存。
