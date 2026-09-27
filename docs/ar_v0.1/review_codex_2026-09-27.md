# AR v0.1 综合 Review（2026-09-27）

## 总体结论

**基础训练与短序列推理可执行，video-first 与 teacher forcing 主干合理，核心因果可见关系通过了本次整模型干预检查；但目前不能认定 v0.1 已按方案完成，也不能认定长时、闭环控制能力已得到验证。**

建议保留主干，处理首帧时间语义并验证训练—推理一致性和自由 rollout。GT-history 结果覆盖已有 `dae9842` 修复提交，尚未完成本次整模型复验；下文保留原始发现用于追溯。

Loss 专项结论：当前尺度不一致，但尚未证明已造成学习失衡（第六节）；推荐验证“受保护尺度校准＋单一 action loss”，暂不增加其他损失项（第七节）。

已阅读 Claude 的“cosmos AR 方案设计”对话、本地方案，并通过 SSH MCP 检查远端代码、提交和运行记录。两轮审阅均未修改远端源码、提交或 push，未启动正式训练，也未干扰现有训练；仅运行独立诊断。

## 审阅范围与快照变化

- 远端仓库：`/mnt/lzh/cosmos-EgoWAM`，分支 `ar-video-action`；工程复测在 Tdebug1，整模型因果干预在 Tdebug4。
- **第一轮快照**：HEAD 为 `6c8e7cafcab2db3fd27b19afbbf8ebbc59f126c3`，相对本地记录的 `origin/ar-video-action` 超前 10 个提交；推理代码 `cosmos3_joint_video_hand_pose/src/ar_inference.py` 与 `tests/test_ar_inference.py` 当时为 untracked。
- **第二轮快照**：Claude 随后提交 `4d491e6`（推理与测试）、`2d10965`（实现文档）。因此推理“尚未提交”的旧状态已失效；提交本身没有修复本文指出的问题。第二轮架构审阅基于 `2d10965`。
- 本次未 fetch、未独立核对服务器端分支，也未在合并文档时重新访问远端；上述状态是各轮已验证快照，不是实时监控结果。
- 第一轮本地与远端 `docs/ar_v0.1_design.md` 的 SHA-256 同为 `85929bb04bf869313d3867b9d61e358eee41eb74aaa002c5cef65d26b9a8f668`。第二轮远端文档已由 `2d10965` 更新，不能继续沿用“当前两端相同”的结论。
- 新版远端实现记录将 KV cache 描述为后续工作，而原方案正文仍要求在 v0.1 中实现；这属于范围与文档一致性问题，需明确确认，不能用实现记录替代验收。
- 下文源码位置均相对于上述远端仓库；省略目录的 `src/`、`ar_model.py` 等项目源码指 `cosmos3_joint_video_hand_pose/src/`。行号对应审阅快照。
- 严重度标签 P1/P2 表示 review 优先级，不是设计文档的 P0–P5 实施阶段。

## 一、架构与方法审阅

### 1. 建模目标：合理，但要明确是哪一种世界模型

令 H_k 为此前的视频、动作及文本条件。现在的分解为：

`p(V_k, A_k | H_k) = p(V_k | H_k) × p(A_k | V_k, H_k)`。

这是一种有效的联合分布分解：先生成视频计划，再根据计划预测动作；不是概率论错误，也不要求把 mask 改成 action-first。当前 chunk 内可以双向去噪，因为推理时整个 chunk 一起生成；它不是逐帧在线因果模型。

但它和 `p(V_k | A_k, H_k)` 的前向动力学接口不同。当前视频分支刻意看不到同一 chunk 的 A_k，所以不能直接输入一个候选 A_k，要求视频分支预测这个动作会导致什么结果。若后续目标是动作候选搜索、反事实仿真或 MPC，应明确增加 action-conditioned 的训练/推理模式，不能仅凭“联合生成”宣称已有此能力。

同样，数据中的 57D 主要是头部/手腕运动与手形轨迹，不是已标定的机器人控制指令。当前 v0.1 可以验证具身表示和生成机制；人体到机器人映射、可达性、接触约束与闭环执行仍是另一层工作。

依据：`src/ar_attention.py:74`、`:119`（均位于 `cosmos3_joint_video_hand_pose/`）；`cosmos3_joint_video_hand_pose/src/action.py:42`、`:143`。

### 2. 注意力计算：核心规则正确，证明范围要分层

实际可见关系：

| Query | 允许读取的 GEN 内容 | 关键禁止项 |
|---|---|---|
| 带噪 V_k | 同块带噪 V_k；更早块的干净 V、A | 干净 V_k、当前 A_k、未来 |
| 带噪 A_k | 同块带噪 A_k；干净 V_k；更早块的干净 V、A | 干净 A_k、未来 |
| 干净 V_k | 当前及更早的干净视频；更早的干净动作 | 当前 A_k、未来、全部带噪流 |
| 干净 A_k | 当前及更早的干净 V、A | 未来、全部带噪流 |

所有 GEN 关系还与 block-id 窗口取交集；真实文本始终可读，padding 被隔离。

代码使用 V_k 的编号 2k、A_k 的编号 2k+1，clean→clean 用 `<=`，noisy→clean 用 `<`，noisy→noisy 用 `==`。**这三个不等式的搭配是关键，当前写法对。** 物理打包顺序虽然是 `[action, vision]`，实际允许关系由 block id 决定，因此不会因此反转生成顺序。

另外检查了文本旁路：`packages/cosmos3/cosmos_framework/model/generator/mot/causal_attention.py:1498` 的 override 分支让 GEN 读取文本，但文本输出调用 `_three_way_text_self_attention`，没有让文本吸收未来视频后再回流到当前块。

两遍 clean/noisy replay 在计算依赖上也是合理的：clean 完全不读 noisy，先算所有 clean K/V，再算 noisy，是该计算图的合法求值顺序。`teacher_forcing_detach_clean_kv=False` 保留跨流梯度，符合这种联合训练计算图。精确等价仍要求位置、timestep、网络参数、随机行为和 mask 全部一致，不能只靠这一原理替代端到端数值测试。

证据边界：上一轮 4 项 GPU 单测证明 FlexAttention 输出和梯度符合稠密参考；但参考复用了相同 predicate，因此它主要验证计算实现，不是独立证明训练没有任何答案泄漏。

#### 本次补做的整模型因果干预

Tdebug4 单卡，Nano checkpoint，真实 T=33 样本，C=4、window=4、sigma=0.4，固定种子 123。每次保持序列长度、文本、位置与其他输入不变，只改变指定 tensor；通过真实模型完整 clean/noisy 前向，比较第一个未来 chunk。正常退出码 0。

| 干预 | 视频预测 relative L2 变化 | 动作预测 relative L2 变化 | 是否符合因果预期 |
|---|---:|---:|---|
| 重复完全相同输入 | 0 | 0 | 是，重复性基线 |
| 扰动后一个 chunk 的所有 clean/noisy 视频与动作 | 0 | 0 | 是，未来未泄漏 |
| 扰动当前 chunk 的干净动作答案 | 0 | 0 | 是，未读取自身动作答案 |
| 扰动当前 chunk 的干净视频答案 | 0 | 1.04365 | 是，动作读取当前视频，视频不读取自身答案 |
| 扰动历史动作 | 0.03754 | 0.21357 | 是，历史动作确实进入后续计算 |

前三项的最大绝对差也为 0。这里扰动幅度较大，非零数值只用于证明计算依赖存在，**不能解释为敏感度优劣、动作理解质量或鲁棒性指标**。这次覆盖一个样本、一组 C/window 和推理 helper 的 teacher-forcing 前向，不替代全部训练噪声路径、完整 rollout 或梯度的验证。

完整记录：Tdebug4 `/tmp/codex-ar-macro-review.jdkqK2/causality.json`；脚本 `check_causality.py`，日志 `causality.log`。

### 3. 已复现的新偏差：首帧 state 的时间位置不对

数据按方案把同一个 t=0 state 重复 K=8 次。但 `packages/cosmos3/cosmos_framework/data/generator/sequence_packing/temporal_causal.py:237` 仅依据 action 行数等于 `latent_t*K`，把这种“整段训练且带真实首状态”的输入当成“每帧都有真实动作的 AR continuation”。

CPU 实测：调用当前 supervised packer，T_latent=3、K=8、fps_video=7.5、fps_action=15，剔除文本偏移后的时间 RoPE 为：

| 内容 | 当前真实 state 输入 | 按首帧同一时刻表达应有的语义 |
|---|---|---|
| V0、V1、V2 | 3.2、6.4、9.6 | 0、3.2、6.4 |
| A0 的 8 个相同 state | 0.4、0.8、…、3.2 | 全部在 t=0 |
| 第一组未来动作 | 3.6、4.0、…、6.4 | 0.4、0.8、…、3.2 |

未来视频与 action 的组内相对对齐仍然正确；整体时间平移本身也未必严重。真正的问题是同一个首状态被编码成一段跨 8 个时刻的历史，且整段训练布局与未来真实增量 cache 入口混为一谈。训练和当前推理共同复用这个路径，所以仅做两者互相比对可能发现不了。

建议优先修正：显式区分“含首帧 state 的完整 clip”和“AR continuation”，首状态统一标在初始时刻，未来动作保留正常时间递增；为这两个入口分别测试绝对起点、相对间隔及 fps 缩放。不应修改 K=8 本身来绕过问题。

### 4. 最大的方法风险：teacher forcing 到自由生成的分布变化

训练视频分支读真实历史；训练动作分支还读当前的真实视频。部署/rollout 时，这两类条件都变为模型自己生成的结果。于是视频误差不仅累积到下一段视频，也传给当前动作；动作误差再进入之后的视频历史。

这是该分解的自然代价，不等于设计错误。但仅训练 loss 下降或 GT-history 预测准确，都不足以证明自由 rollout 稳定。LingBot 的公开训练代码确实启用了 50% 概率的视频条件扰动，而当前配方关闭了这类增强；不能把“mask 对齐 LingBot”理解为整个训练 recipe 等价。

建议先做可解释的条件消融：

1. GT 历史 + GT 当前视频：动作分支能力上限。
2. GT 历史 + 生成当前视频：隔离视频计划误差对动作的影响。
3. 生成历史 + 生成当前视频：实际自由 rollout。

三者都看逐 chunk 误差、末端漂移及视频—动作一致性。若 1 好、2 差，优先解决当前视频条件鲁棒性；若 2 好、3 差，重点解决历史误差累积。先以条件扰动作为较小改动的消融，再考虑训练时引入自身 rollout，不必立即重构整个训练算法。

依据：`cosmos3_joint_video_hand_pose/src/ar_model.py:265`、`ar_inference.py:145`；[LingBot 训练实现](https://github.com/Robbyant/lingbot-va/blob/main/wan_va/train.py)。训练/推理历史分布差异也是 [Self Forcing](https://arxiv.org/abs/2506.08009) 所讨论的问题，但该论文不直接证明本项目必须采用它的训练方法。

### 5. 遮挡动作：是扩展数据时的合同风险，目前子集影响很小

`loss.py:78` 对不可见手部置零直接预测 loss；但 `ar_dataset.py:196` 仍构造完整动作向量，clean attention 未使用可见性屏蔽这些历史值。模型训练可读取不可见手部的 GT，推理却必须读取生成值。缺少直接监督不代表没有任何间接梯度，但也不能保证这些生成坐标可靠。

为了判断现实影响，读取了 148 个可用 segment 的完整源帧（不是随机训练窗口加权统计）：78,148 行中，右手只有 1 行视野标记为不可见，左手为 0。因此**这不是目前 36 episode 子集的主要故障解释**；不要为几乎未出现的遮挡立即大改模型。换数据集前应解决。

未来需明确：FOV 是否表示“不值得监督”还是“标签不可信”？视野外但跟踪可靠的手部，可以继续监督；标签确实缺失时，应设计一致的缺失值/有效性表示，使训练和推理都遵守。不能用未来 GT 可见性在推理时作弊屏蔽。增量还依赖前一帧姿态，可见性与跟踪有效性不能混为一谈。

### 6. 时间采样与动作表示：大体合理，但别过度解释

- **K=8 是对的**：源视频 30fps、stride=2 后视频 15fps，VAE 时间压缩 4，4 个视频帧覆盖 8 个源时刻。保留 30Hz 动作对应 8 个 token。fps_video=7.5、fps_action=15 同比缩放，相对钟速仍一致。
- **fps 减半不是“每步动作减半”**：mRoPE 确实按 fps 缩放，标签会拉长模型时间，按更低频率回放也会变慢；它没有改变归一化前的相邻动作增量，更没有解决人体/机器人运动范围、动力学与本体差异。speed_factor=1 应作为后续对照，而不是默认认定 0.5 更利于机器人迁移。
- **B3 局部增量适合短期运动学习**：比跨长时间的首帧差分更局部。但解码通过 SE(3) 逐步积分，微小偏差会累积；因此不能只看每步归一化 MSE，应看逐 chunk 末端位置/朝向漂移。有限窗口下，初始锚点最终不可直接读取，未来应评估边界 state 或真实观测刷新。
- **57D 占位合理，可插拔接口尚未实现**：保持 codec 不变适合先验证 AR；现有 AE15 手形并不是 21 DoF 的完整机器人控制合同。

依据：`cosmos3_joint_video_hand_pose/src/ar_dataset.py:45`、`:72`；`packages/cosmos3/cosmos_framework/data/generator/sequence_packing/mrope.py:165`；`cosmos3_joint_video_hand_pose/src/action.py:64`、`:105`。

### 7. Chunk、窗口、loss 与计算预算

- C∈{1,2,3,4} 训练、C=4 推理，是合理的长度鲁棒性选择；每个 chunk 一个 sigma、视频/action 独立 sigma，也与当前分块 solver 相容。LingBot 公开代码按 latent 帧采样噪声；这里按 chunk 采样是有意的变体，不能称为逐项照搬。
- 文本与 GEN 合在一个 softmax，适配的是 Cosmos 的注意力路径；LingBot 的文本 cross-attention 结构并不完全相同。可以借用因果规则，不应把整个网络称为严格等价实现。
- window=30 的单位是视频/动作 block-id，不是 30 视频帧。当前 T=129、C=4 只有 8 个未来 chunk，窗口基本不截断。短片段成功不能证明长时 cache 淘汰正确。LingBot `create_empty_cache` 按 `window//2` 份视频和动作 chunk 容量分配，可作为实现语义参照。
- C=4 对应 32 个动作：原始时间约 1.07 秒、减速后的模型时间约 2.13 秒。离线生成合理；闭环控制是否执行整块、多久接收新观测必须另外设计。当前两段串行去噪且无持久 cache，也没有实时性的证据。
- video 1.0、action 0.7 是合理起点，但不等于梯度贡献比例。当前按物理子块等权避免 15D 手形仅因维数大压过 3D 平移，这是有理由的选择。应记录分块物理误差和模态梯度贡献再调权重。
- 单卡每步一个 clip、8 卡约 8 clip/step（无额外累积时），与原来多样本 packing 的 batch 不同。固定 1200 steps 不自动构成与旧版本公平的训练预算比较；应同时报告 clip/token 曝光量、GPU 时间和有效更新次数。

依据：`cosmos3_joint_video_hand_pose/src/ar_model.py:107`、`:127`、`:161`；`src/config.py:324`、`:349`、`:363`；`src/loss.py:96`。上述 `src/` 均属于 `cosmos3_joint_video_hand_pose/`。[LingBot mask / cache 代码](https://github.com/Robbyant/lingbot-va/blob/main/wan_va/modules/model.py)支持其窗口与可见性规则；本项目的优劣判断是本次审阅推论。

## 二、实现与验收问题

### 1. [P1] GT-history 评测会用真值覆盖已生成的预测

位置：`cosmos3_joint_video_hand_pose/src/ar_inference.py:157–159`，下游评测在 `:337–345`。

`sample(history='gt')` 在生成每个 chunk 前，会把同一份输出缓冲区的历史视频和动作覆盖为 GT。这样做可以为当前 chunk 提供真实历史，但同时丢失了前面 chunk 的预测。函数最终返回的序列中，只有最后一个 chunk 保留预测，其余未来 chunk 已被替换成真值；随后却对整个未来序列计算动作指标和视频 PSNR、导出回放。

实测复现：使用 9 个 latent、C=4、K=8，GT 全部设为 100，forward 替换为固定速度预测。运行真实 `ARSampler.sample()` 后：第一个生成 chunk 的视频和 action 与 GT 完全相等，最后一个 chunk 不相等。该测试无需模型权重，直接验证了缓冲区覆盖。

影响：真实历史条件下的评测结果失真，不能用于判断逐 chunk 预测质量。对 T=129 的 8 个未来 chunk，前 7 个都会被覆盖；默认 `history='generated'` 不触发这个特定问题。

建议：分离“模型输入历史”和“累计预测输出”，每个 chunk 生成后单独保存预测；指标只在保存的预测上计算。补充两 chunk 以上的回归测试，断言早期预测不会被 GT 替换。

### 2. [P2] 一致性检查不是可判定通过/失败的验收测试

位置：`cosmos3_joint_video_hand_pose/src/ar_inference.py:174–221`、`:333–335`。

检查仅返回 relative L2 数字，没有容差、有限值断言或失败退出逻辑。因此退出码 0 只说明脚本运行结束，不能解释成“训练与推理数值一致”。

Claude 最后一次结果位于 `/tmp/egowam/consistency_nano/consistency.json`：2 个 T=65 样本中，非末尾 chunk 的视频相对 L2 约 1.63%–1.88%，动作约 0.90%–1.14%；末尾 chunk 为 0。仅凭这些数字，不能直接断言误差都是 bf16 引起的，也不能断言一定存在 mask 错误，需要精度对照和容差依据。

覆盖也不完整：它用同一个 `forward()` 比较全长和截断序列，且视频/action 共用 sigma，未覆盖训练路径中独立 action 加噪和 timestep 覆写（`ar_model.py:161–223`）。

建议：建立带阈值和 NaN/Inf 检查的测试，比较真实训练打包/加噪路径与推理路径，覆盖独立 sigma、多个 chunk 和有限窗口。误差超阈值应返回失败。

### 3. [P2] 当前推理未实现方案要求的持久 KV cache

位置：`cosmos3_joint_video_hand_pose/src/ar_inference.py:4–10`、`:115–136`、`:156–169`。

每次 solver step 都重新构造并执行从序列开头到当前 chunk 的 teacher-forcing 前向，文件说明也明确写了“不需要独立 KV cache”。这可以作为短序列参考实现，但不是方案第 6 节约定的“生成完刷新干净 KV，下一 chunk 复用”的实现。窗口 mask 限制可见范围，并未使历史 token 的打包和计算长度固定。

影响：不能将短片段运行成功视为有界缓存的长 rollout / 流式推理完成；历史增长时仍会重复计算。方案中的 `image2video_joint_action` 入口也尚未由此脚本兑现。

建议：保留该实现作参考路径，补充 cache prefill、视频/动作 clean refresh、窗口淘汰及与参考路径的一致性测试；若要暂缓 cache，需要明确修改并重新确认方案范围。

### 4. [P2] 当前 smoke 不能验证保存恢复，也不足以证明梯度健康

位置：`cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_1_smoke.sh:2`、`:22`；`src/smoke_train.py:117–171`。

入口默认仅 4 步，并明确不写 checkpoint。结果检查包含 loss 与动作子块 loss 的有限值，但输出记录不包含方案要求的原始梯度 NaN/Inf 计数；也没有完成一次保存后重新加载继续训练的验证。

已有结果 `/mnt/lzh/cosmos-EgoWAM/outputs/joint_video_hand_pose/ar/smoke/smoke_ar_v0_1_20260927_081049/smoke_result.json` 为 `success`、8 卡、4 步。这确实证明基础训练能执行，但不足以满足方案第 8 节的“8 卡 50 步、原始 NaN/Inf 为 0、checkpoint 保存恢复”。仅将 STEPS 改为 50 也不会补齐保存恢复测试。

建议：补充独立验收模式，采集原始梯度有限性，执行保存、退出、恢复和下一步训练，并核对 iteration、优化器及数据恢复状态。

## 三、工程复测结果

所有诊断产物均写入独立目录 `/tmp/codex-ar-v01-review.abv6O9`，没有覆盖 Claude 的实验输出。

| 项目 | 结果与边界 |
|---|---|
| 仓库 `tests` CPU 回归 | 111 passed、4 skipped，19.30 秒；见 `pytest.log`、`pytest.exit` |
| AR attention GPU 回归 | 4 passed，28.95 秒；见 `ar_gpu.log`。补跑了 CPU 回归中跳过的这 4 项 |
| GT-history 缓冲区覆盖 | 最小复现确认存在，见发现 1 |
| 单卡真实生成 | Tdebug1 GPU2，Nano `iter_000048464/model`，T=33、C=4、视频和 action 各 2 步、1 个样本、generated history；退出码 0 |
| 生成链路覆盖 | 权重加载、数据/VAE 编码、两个未来 chunk 生成、动作指标、VAE 解码、两种帧率 MP4 导出均完成 |
| 质量解释 | 未经过本次 AR 正式训练，且仅 2 步采样；本次仅验证执行链路，不用于评价收敛或最终效果 |
| 官方框架单测复跑 | 当前环境缺少测试插件：先报未知 `--suppress-no-test-exit-code`，移除默认附加参数后报未知 `pytest_xdist_auto_num_workers` hook；未安装依赖或修改配置，不能声称本次复跑通过 |

真实生成日志：`generation_retry.log`、`generation_retry.exit`。结果在 `generated_retry/`：`results_generated.json`、`0001_generated.npz`、`0001_generated_model_time.mp4`、`0001_generated_real_time.mp4`。

第一次在 GPU0 的生成尝试因另一进程同时占用显存而 OOM；进程退出后换空闲 GPU2 成功。不将这次资源冲突列为代码缺陷。另有短暂 SSH 连接超时，直接重试恢复，不能据此认定 MCP 不可用。

## 四、其余方案缺口

- `src/codec.py` 目前仍是具体的 `FrozenHandMLPAE15`，数据/推理直接使用 57D 的 `Action57Builder`；方案要求的统一可插拔 `ActionCodec` 尚未完成。
- 当前推理导出 GT/预测左右视频，不等于方案要求的完整四格回放与固定 4 样本单 chunk/整段 rollout 验收。
- 正式训练 launcher `launch_ar_v0_1.sh:20` 拒绝已有任务目录，若需要断点续训，应另行明确恢复入口，不能直接假定原命令可重跑恢复。

## 五、统一后续顺序

1. **先修确定问题**：分离 GT-history 输入与预测输出；显式区分含首帧 state 的完整 clip 与 AR continuation，修正首帧时间布局，并补充回归测试。
2. **保留并扩展因果验证**：本次整模型干预已在一个样本、一组 C/window 下通过，后续覆盖更多 chunk、窗口边界和训练路径，不必把已完成测试当作尚未开始。
3. **建立可判定的一致性验收**：比较真实训练打包/加噪与推理路径，覆盖独立 action sigma、NaN/Inf、精度基线及失败阈值。
4. **完成训练可靠性检查**：八卡 50 步、原始梯度有限性、checkpoint 保存—退出—恢复—下一步验证；明确正式 launcher 的恢复入口。
5. **评估自由生成**：同权重比较“GT 历史＋GT 当前视频”“GT 历史＋生成当前视频”“生成历史＋生成当前视频”，记录逐 chunk 误差、末端漂移及视频—动作一致性，不能只用全片平均值。
6. **明确 cache 范围并落实**：补齐持久 KV cache，与完整历史参考路径逐步对照，特别验证窗口首次淘汰；不能简单截掉历史再重算，假定与保留历史形成的多层 KV 等价。若暂缓，需确认方案范围并统一正文与实现记录。
7. **再解释正式实验结果**：需结合上述验收判断结果，而不是仅看 1200 steps 是否结束。后续逐项比较条件扰动、speed_factor、窗口等因素；36 episode 训练集过拟合只证明机制和容量，不证明泛化。
8. **同步实现与文档**：推理代码和实现文档已由 Claude 提交；未来修复应记录独立提交和测试证据。push 仍按用户另行授权执行。

诊断产物保留在远端独立目录：Tdebug1 `/tmp/codex-ar-v01-review.abv6O9/`，Tdebug4 `/tmp/codex-ar-macro-review.jdkqK2/`。本文仅合并审阅记录，不代表上述修复已实施。

## 六、Loss 审阅：尺度不一致，尚未证明学习失衡

**已确认：当前各部分的数值尺度、隐含权重和加噪信噪比不同。尚未确认：这些差异是否已经造成某部分欠拟合、另一部分过拟合。分块计算不是根因。**

### 当前目标与隐含权重

57D action 包含相机位姿 9D，以及左右手各 24D（平移 3D＋旋转 6D＋手形编码 15D）。每只手的 20 个关键点已压缩成一个 15D 编码，并非逐关节设置 loss。

当前对相机、左右手的平移／旋转／手形共 8 块分别取 MSE，再对有效块取平均：

`L_total = L_video + 0.7 L_action`

这里的 MSE 回归 flow 向量 `ε−x`，不是直接回归物理位置；条件状态、padding 和不可见手部不计入相应监督。

所有块有效时，每块权重为 1/8，因此单个平移坐标的系数是单个手形坐标的 **5 倍**。这等价于一个整体加权 MSE，数值及梯度等价检查已通过。**块等权不等于梯度等大；改成整体 MSE，也只是改为坐标等权。**

### 实测尺度与风险

当前 normalizer 下，148 个训练 segment、78,000 行未来动作的统计如下（各通道标准差的块内平均）：

| 表示 | 相机 | 右手 | 左手 |
|---|---:|---:|---:|
| 平移 | 0.805 | 0.914 | 0.832 |
| 6D 旋转 | 0.350 | 0.347 | 0.323 |
| 手形编码 | — | 0.996 | 1.112 |

原因是平移、旋转、手形分别使用标准差尺度、分位数尺度和 codec 自带统计量。当前 `shift_action=5`，噪声强度 σ 的中位数约 0.83；同一 action chunk 内各坐标共享 σ。

相同 σ 下，边际信噪比正比于数据方差。因此，**现有归一化没有消除尺度差异，块等权也不能修正输入的加噪比例。** 但这些数据变化不全是标注噪声，模型还有视频和历史条件，不能仅据此断言旋转没学好。

统计是完整 segment 的帧加权结果，不是实际训练窗口曝光分布。是否学习失衡，需结合分部位的训练／验证物理误差和共享参数梯度判断。

证据：远端快照 `dae9842`；`src/loss.py:8、84–105`、`src/normalization.py:17–42`、`src/codec.py:18–33`、`src/ar_model.py:162`；正式训练实际保存配置。CPU 诊断成功，产物保留于 Tdebug4 `/tmp/codex-ar-loss-review.RHNbw2/`（脚本、日志及 `stats.json`）。

## 七、推荐方案：尺度校准＋一个 action loss

**保留 57D 表示，先校准进入模型的数据尺度，再使用一个整体 masked Flow Matching MSE。分部位指标只用于诊断，不增加训练损失。方案待验证，尚未实施。**

### 改什么

1. **固定尺度校准**：仅用训练集有效数据拟合均值与尺度，采样尽量匹配训练窗口；state／future 分开处理，训练与推理共用同一版本。
2. **限制过度放大**：正常通道校准到相近量级；近常量或不可靠通道设置尺度下限、限制放大，并检查原归一化是否已放大噪声。不强行让所有通道方差等于 1，不用硬截断丢弃大动作。具体保护阈值需由数据检查确定。
3. **整体计算 loss**：在校准后的 z 空间加噪、预测和积分，最后反归一化。对每个样本直接平均所有有效坐标的误差，再平均样本：

`zσ = (1−σ)z + σε`

`L_action = mean_valid[(v_pred − (ε−z))²]`

总目标暂保持 `L_video + 0.7 L_action`。不再先对八块分别归一分母，不新增动态权重、tokenizer 或几何辅助 loss。

**意义与代价**：校准发生在加噪前，能同时调整数值尺度和边际信噪比；但不能消除标注噪声或保证梯度均衡。坐标等权后，手形占 30/57 个坐标，即 52.6%，不再是当前的 25%。这是明确的权重选择，不是“完全没有权重”。

### 开源依据

- **EgoVerse／EgoWAM** 的 action 分支使用整体 MSE，支持“归一化＋单一动作目标”的路线。[EgoVerse 代码](https://github.com/GaTech-RL2/EgoVerse/blob/main/egomimic/models/denoising_policy.py)、[EgoWAM 固定快照](https://github.com/GaTech-RL2/EgoWAM/blob/c87617fe37a6ed6a951e6b176ad552200c425c93/egowam/models/denoising_policy.py#L85)。
- **EgoWAM** 对分位数尺度设置下限，防止近常量通道放大异常；同时对裸 z-score 给出稳定性警告。可借鉴保护原则，不能照搬其 14D 动作配方或阈值，也不能声称它验证了本方案。[归一化保护](https://github.com/GaTech-RL2/EgoWAM/blob/c87617fe37a6ed6a951e6b176ad552200c425c93/egowam/rldb/zarr/utils.py#L664)、[稳定性警告](https://github.com/GaTech-RL2/EgoWAM/blob/c87617fe37a6ed6a951e6b176ad552200c425c93/egowam/algo/hpt.py#L751)。

以上结合 PaperAutoRead 已有材料与官方代码核对；推荐的受保护尺度校准是本项目候选，不是论文已证实的结论。

### 怎么验证

| 对照 | 归一化 | Action loss | 要回答的问题 |
|---|---|---|---|
| A：现有基线 | 当前方式 | 八块等权 | 当前表现 |
| B：只改 loss | 当前方式 | 整体坐标等权 | 权重变化是否有益 |
| C：推荐候选 | 受保护尺度校准 | 整体坐标等权 | 尺度校准是否进一步有益 |

先检查尺度稳定性、mask 和反归一化往返，再做固定数据、预算与初始化策略的对照。**本轮不同时改噪声采样**；新尺度与旧 action checkpoint 的适配必须先明确。

以验证集手腕位置、旋转角、指尖误差和自由 rollout 漂移验收，并结合训练集曲线识别过拟合；按噪声强度分桶、低频抽查共享参数梯度。**不直接比较不同归一化下的总 loss 数值。**
