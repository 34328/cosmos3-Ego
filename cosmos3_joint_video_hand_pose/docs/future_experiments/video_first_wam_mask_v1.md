# Video-First WAM Modality-Causal Mask 实验方案 v1

> 状态：后续实验评审稿；不属于 overfit_v0.0 当前实现合同  
> 范围：EgoVerse Cosmos 3 Nano，joint video + camera/hand action WAM  
> 核心原则：第一轮只改变 Generator 内 GEN attention 的模态可见性，不改变数据、loss、noise schedule、参数训练边界或采样器。

## 1. 目标

当前 joint WAM 在同一个 Cosmos Generator 中同时对 future video latent 与 future action 做 flow matching。现有 two-way attention 中，每个 GEN query 都可以读取同 sample 的全部 UND/GEN key，因此 future video 与 future action 在每层都是双向耦合的。

本实验不尝试一次性把 Cosmos 改造成两阶段模型，而是加入一个最小的 video-first 模态三角 mask，使单次联合去噪满足：

$$
\hat v_V = f_V(C, x_V(t), t;\theta),
$$

$$
\hat v_A = f_A(C, x_V(t), x_A(t), t;\theta),
$$

其中：

- $C$ 包含 text、首帧视频 $I_0$ 和首帧真实 action state $a_0$；
- $x_V(t)$ 是当前去噪时刻的 future video latent；
- $x_A(t)$ 是当前去噪时刻的 future action；
- video velocity 不读取 future action；
- action velocity 可以读取当前 video 去噪状态。

该设计优先保护视频生成路径，同时让 action 通过 video hidden states 学习与生成视频一致的 camera/hand trajectory。

本方案建立的是生成图上的定向条件依赖，不宣称从观测数据中识别了物理世界因果。对于当前任务，其目标生成顺序是：

~~~text
text + I0 + a0
       |
       v
 future video
       |
       v
 future camera/hand action
~~~

## 2. 当前实现事实

### 2.1 Joint 工程没有修改 Generator 架构

EgoVerseOmniMoTModel 当前只是 Cosmos OmniMoTModel 的 loss adapter，主要增加 visibility-weighted action flow loss 和日志；Generator attention 仍完全使用 packages/cosmos3：

- cosmos3_joint_video_hand_pose/src/model.py:35-43
- cosmos3_joint_video_hand_pose/src/model.py:67-115
- cosmos3_joint_video_hand_pose/src/model.py:117-163

因此 mask 不应放在 loss adapter 外部，而应进入 Cosmos3VFMNetwork 构造 attention metadata 的位置。

### 2.2 当前 WAM 条件和目标

当前 Case-B WAM 布局为：

~~~text
vision condition: [I0]
vision target:    [I1, ..., IT-1]

action condition: [a0]
action target:    [a1, ..., aT-1]
~~~

对应实现：

- condition_frame_indexes_vision=[0]：packages/cosmos3/cosmos_framework/data/generator/action/transforms.py:283-290
- action_length=video_length 时 condition_frame_indexes_action=[0]：packages/cosmos3/cosmos_framework/data/generator/action/transforms.py:320-357
- inference 显式校验 a0 为 clean condition：cosmos3_joint_video_hand_pose/src/inference.py:48-60

### 2.3 Packer 已经保留构造 mask 所需的信息

PackedSequence 中分别保留：

- vision.sequence_indexes
- vision.condition_mask
- vision.noisy_frame_indexes
- action.sequence_indexes
- action.condition_mask
- action.noisy_frame_indexes

对应实现：

- vision：packages/cosmos3/cosmos_framework/data/generator/sequence_packing/modalities.py:283-316
- action：packages/cosmos3/cosmos_framework/data/generator/sequence_packing/modalities.py:393-424

不需要修改数据集或给模型增加新的输入字段。

### 2.4 当前注意力是双向 GEN full attention

two_way_attention 的 dense GEN 分支使用 full queries 对 sample 内全部 UND/GEN key/value 做 attention：

- packages/cosmos3/cosmos_framework/model/generator/mot/attention.py:297-311

Cosmos 已经存在可复用的 FlexAttention 入口：

- SplitInfo.flex_block_mask：packages/cosmos3/cosmos_framework/model/generator/mot/attention.py:72-83
- two_way_attention Flex 分支：packages/cosmos3/cosmos_framework/model/generator/mot/attention.py:272-296
- network 每个 forward 构造一次 mask 并供所有 decoder layers 复用：packages/cosmos3/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1100-1126

现有缺口是该路径只支持 vision-only multiview batch，并显式拒绝 action：

- packages/cosmos3/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py:1082-1095

## 3. Mask 的精确定义

### 3.1 Token role

对每个 sample 的 GEN token 定义三个 role：

- C：clean condition，包括 $I_0$ 的全部空间 token 和 clean action state $a_0$；
- V：future noisy video token；
- A：future noisy action token。

UND/text 保持为独立 causal stream。

### 3.2 可见性矩阵

| Query \ Key | UND | C | V | A |
|:---|:---:|:---:|:---:|:---:|
| C | yes | yes | no | no |
| V | yes | yes | yes | no |
| A | yes | yes | yes | yes |

所有规则还必须满足 same_sample。真实 token 不得读取其他 packed sample 或 padding；padding query 只读取 padding，避免 empty-softmax NaN。

等价 role rank 为：

~~~text
C = 0
V = 1
A = 2

GEN-to-GEN allowed iff:
same_sample AND key_role <= query_role
~~~

GEN query 仍可读取同 sample 的全部 UND key。UND query 继续只执行原有 causal UND attention，永远不读取 GEN。

### 3.3 为什么 C query 也必须被限制

只屏蔽 V-query -> A-key 不够。

如果 clean condition query 可以读取 future action，那么多层 Transformer 中会出现间接路径：

~~~text
layer k:     future A -> updated C
layer k+1:   updated C -> future V
~~~

因此 C query 只能读取 UND 和 clean C。这样才能保证在任意层数下，future action 都不能经 condition token 绕回 video。

### 3.4 该 mask 不包含时间因果

V1 只做模态因果，不额外加入 frame-wise causal mask：

- future V token 仍可读取同 clip 的全部 future V；
- future A token 可读取同 clip 的全部 V 和 A；
- 时间对齐继续由现有 mRoPE 和训练数据承担。

不要在同一实验中同时加入 A_t 只能读取 V_<=t 等时间限制，否则无法判断收益来自模态方向还是时间方向。

## 4. 共享梯度的准确含义

### 4.1 Action loss 更新共享主干是预期行为

第一版必须保留 action loss 对 moe_gen/time_embedder 等共享生成参数的反向传播。

mask 要保证的是固定参数、同一次 forward 内：

$$
\frac{\partial \hat v_V}{\partial x_A(t)}=0,
\qquad
\frac{\partial \hat v_A}{\partial x_V(t)}\ne0.
$$

action loss 通过 A-query -> V-key/value 路径更新共享表示，是模型学习 video/action 对齐的主要机制，不是反向信息泄漏。

优化一步后共享参数 $\theta$ 发生变化，从而可能改变下一次 forward 的 video 输出，这是正常的联合学习，不违反上述结构约束。

### 4.2 V1 明确不做

- 不 detach video hidden states；
- 不阻断 action loss 到共享主干的梯度；
- 不冻结 moe_gen/time_embedder；
- 不给 action 单独增加 Transformer；
- 不修改 video/action loss 权重；
- 不启用 independent_action_schedule；
- 不改变 action representation、normalizer 或 codec；
- 不修改 UniPC/采样步数。

梯度隔离只能作为后续独立 ablation，不能混入本轮 mask 验证。

## 5. 同步 flow matching 下的含义和边界

当前配置 independent_action_schedule=False，同 sample 的 video/action 共用 sampled sigma：

- cosmos3_joint_video_hand_pose/src/config.py:68-77
- packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py:1798-1811

因此 action 在每个 denoise step 读取的是当前时刻 $x_V(t)$，不是已经完全解码的 clean final video。

加入三角 mask 后，联合 ODE/flow vector field 具有三角结构：

~~~text
dxV/dt = fV(C, xV, t)
dxA/dt = fA(C, xV, xA, t)
~~~

video 轨迹独立于 action 轨迹；action 轨迹跟随 video 去噪轨迹共同演化。只要训练与 inference 每个 denoise call 都使用同一 mask，这就是合法的 simultaneous triangular flow。

它比完整的“两次生成：先得到 clean video，再做 inverse dynamics”更小，但也不等价于完整两阶段。若 V1 表明方向性有效而 action 仍跟不上 video，下一步才单独评估 video-leading sigma 或两阶段 sampler。

## 6. 最小实现设计

### 6.1 Feature flag

建议增加独立配置，而不是复用含义为 multiview 的 enabled：

~~~text
generation_attention_mask: dense | full_flex | video_first_wam
~~~

第一版也可以用两个 bool，但枚举可以避免同时开启两种 mask。

含义：

- dense：完全保持当前 two_way_attention 路径；
- full_flex：使用 FlexAttention，但 GEN 可见性与 dense same-sample full attention 语义一致；
- video_first_wam：使用本文的 C/V/A 三角 mask。

full_flex 是必要的实验控制组，原因见第 8 节。

### 6.2 Mask builder

在 packages/cosmos3/cosmos_framework/model/generator/mot/flex_attention.py 增加 WAM 专用 metadata/builder，例如：

~~~text
build_wam_flex_metadata(...)
build_wam_block_mask(mask_mode="full" | "video_first")
~~~

metadata 至少包含：

- sample_id
- role_id: C/V/A
- is_und
- padding sentinel
- num_und

vision condition mask 需要从 [T,1,1] 展开到每帧的 H*W patch token；action condition mask 为每帧一个 token。

不要依赖“vision 一定在 action 前”的裸区间假设作为唯一真相。应从 token_shapes、condition_mask 和 pack offsets 构造 role metadata，并对实际长度做显式断言。

### 6.3 Network 接入

在 Cosmos3VFMNetwork.forward 中：

1. build_packed_sequence 时按选定 Flex backend 的 alignment 对 GEN/UND stream padding；
2. 在 decoder layers 外构造一次 WAM BlockMask；
3. 写入 attention_meta.flex_block_mask；
4. 写入 attention_meta.flex_backend；
5. 所有层继续复用现有 two_way_attention Flex 分支。

训练与 inference 都调用同一个 network forward，因此不能只在 sampler 外层临时加 mask。

### 6.4 V1 fail-fast 范围

video_first_wam V1 只接受：

- joint_attn_implementation=two_way；
- vision 与 action 同时存在；
- sound 不存在；
- video_temporal_causal=False；
- multiview FlexAttention 未开启；
- NATTEN 未开启；
- context parallel size=1；
- compile disabled；
- 每个 sample 为当前 EgoVerse Case-B WAM，即 I0/a0 clean。

该后续实验若实施，必须按当时选定的并行拓扑重新验证；不能把历史 CP 假设当作当前事实。

### 6.5 Backend 固定

第一版建议显式固定：

~~~text
flex_attention.backend=triton
~~~

不要使用 auto，以免不同节点是否安装 FlashAttention-4 导致 backend、padding 和数值舍入变化。官方配置说明也指出需要可比运行时应固定 backend：

- packages/cosmos3/cosmos_framework/configs/base/defaults/flex_attention.py:74-85

## 7. 正确性测试

### 7.1 Mask truth-table 测试

在小型 packed batch 上构造 dense boolean reference，逐项验证：

- C -> C 可见；
- C -> V/A 不可见；
- V -> C/V 可见；
- V -> A 不可见；
- A -> C/V/A 可见；
- GEN -> 同 sample UND 可见；
- 跨 sample 全部不可见；
- real/padding 隔离；
- padding query 至少有 padding key。

### 7.2 Full-Flex 等价性测试

full_flex 必须与原 dense two_way attention 在同一输入上满足：

- forward 输出 allclose；
- query/key/value gradients allclose；
- 多 sample、variable length、padding 都覆盖。

现有 flex_attention_test.py 已经包含 dense reference、block-mask 和 fused gradient 测试框架，可沿用：

- packages/cosmos3/cosmos_framework/model/generator/mot/flex_attention_test.py:325-373
- packages/cosmos3/cosmos_framework/model/generator/mot/flex_attention_test.py:1266-1291

### 7.3 Video 对 action 输入的干预不变性

固定：

- text、I0、a0；
- video xV(t)；
- timestep；
- 参数。

只替换 future action noise/xA(t)，要求：

~~~text
preds_vision_before == preds_vision_after
~~~

使用 bf16 合理容差，但误差不得随 action 扰动幅度增长。

### 7.4 Action 对 video 输入的敏感性

固定 text、I0、a0、xA(t)，改变 future video xV(t)，要求：

~~~text
preds_action_before != preds_action_after
~~~

这不是要求随机初始化时达到高语义相关，只要求计算图中存在 V -> A 的有效路径。

### 7.5 两层 condition-bridge 泄漏测试

必须至少通过两个 attention/decoder block 后再做 7.3 的不变性测试。

单层测试无法捕获：

~~~text
A -> C at layer 1
C -> V at layer 2
~~~

该测试专门防止错误实现只屏蔽 V-query -> A-key，却遗漏 C-query -> A-key。

### 7.6 Gradient 路径测试

分别验证：

1. video output 对 future action input 的梯度为零；
2. action output/loss 对 video token hidden states 的梯度非零；
3. 只反传 action loss 时，共享 moe_gen 参数存在有限非零梯度；
4. 不出现 NaN/Inf 或 all-masked real query。

第 2、3 项是期望的跨模态学习机制，不能被误判为泄漏。

### 7.7 Mask-off 回归

generation_attention_mask=dense 时，不构造 BlockMask，不改变当前 attention path。现有 checkpoint 的训练和 inference 行为不应因新代码而变化。

## 8. 实验对照设计

### 8.1 为什么不能只比较 Dense 与 Triangular-Flex

如果直接比较：

~~~text
J0: dense attention
J1: FlexAttention + triangular mask
~~~

则同时改变了：

- attention 可见性；
- attention kernel；
- stream padding/alignment；
- 数值舍入和性能特征。

质量差异不能完全归因于因果 mask。

### 8.2 必要的同 backend 控制组

正式 mask-only 比较应为：

~~~text
J0-flex-full:
    FlexAttention
    same-sample full GEN visibility

J1-video-first:
    FlexAttention
    C/V/A triangular visibility
~~~

两组使用同一 backend、同一 padding、同一初始化和训练配置，只改变 mask predicate。

原 dense 模型保留为实现回归参照，不需要第一轮重复跑完整 300 steps；先通过 full_flex 与 dense 的 forward/backward 等价测试。

## 9. 分阶段运行计划

### Phase A：只做代码正确性

目标：

- 完成 feature flag 和两个 WAM Flex mask；
- 通过第 7 节全部单元测试；
- 不启动训练；
- 不改 overfit_v0.0 checkpoint 或输出。

退出条件：full_flex 与 dense 等价，video-first 的双向干预测试符合预期。

### Phase B：8 卡 3-step smoke

J0-flex-full 和 J1-video-first 各运行 3 steps：

- 真实 EgoVerse packed batch；
- FSDP=8、CP=1；
- forward/backward/optimizer step；
- 记录 peak memory、tokens/s、step time；
- 检查 loss、gradient、BlockMask 行数和实际 role token 计数。

退出条件：无 OOM、NaN、deadlock、empty attention row，两个实验读取相同 sample ids。

### Phase C：100-step paired screening

初始化：

- 优先从质量已验收的 IT2V step 300 或 600 完整 DCP 初始化；
- J0/J1 使用同一个 checkpoint；
- load_training_state=false；
- optimizer/scheduler/iteration 从零开始。

两组保持完全一致：

- 36 episodes / 181 segments；
- structured prompt；
- video/action loss 10:7；
- base LR 2e-5；
- action projection 5x LR；
- independent_action_schedule=False；
- seed、sampler、初始 noise 和 replay inputs；
- UniPC 30 steps、shift=5。

必须记录每个 optimizer step 的 sample_id 序列，并在 J0/J1 之间校验一致。多 worker iterable dataset 仅有相同 seed 不足以证明实际数据顺序相同。

在 step 100 生成同一组固定四格 replay：

- 生成视频上的预测双手投影；
- 预测 F0 第三人称轨迹；
- GT 视频上的 GT 投影；
- GT F0 第三人称轨迹。

### Phase D：300-step confirmation

只有 Phase C 满足以下条件才继续：

- video-first 没有明显降低视频质量；
- action 没有数值发散；
- 预测手投影与生成画面的对齐有正向信号；
- 训练速度和显存成本可接受。

J0/J1 各跑到 300，使用同 seed 固定 replay 做盲评。

## 10. 观测指标

### 10.1 结构正确性

- action 扰动下 video prediction 最大绝对差；
- video 扰动下 action prediction 差异；
- action loss 到 shared/video representation 的 gradient norm；
- 每种 role 的 token 数；
- 每种 query-role 到 key-role 的允许 block 数。

### 10.2 Video

- loss/video_raw 只作为训练稳定性指标；
- 固定 seed replay 的清晰度、时间一致性、首帧保持和动作完成度；
- J0/J1 盲评，不依赖单次 loss 高低判断画质。

### 10.3 Action

- 现有 camera/wrist/fingertip trajectory metrics；
- action_raw 及八个 action 子块 raw loss；
- 长时间段是否出现漂移、手型坍缩或左右手交换。

### 10.4 Video/action alignment

优先使用项目已经实现的预测 action 投影到生成视频的第一行 replay：

- wrist/hand projection 是否落在生成手上；
- camera motion 与生成视角变化是否一致；
- action 发生变化的时刻是否与视频事件一致；
- action 轨迹是否平滑但错误地追随 GT，而不是追随生成视频。

正式 replay 合同见：

- cosmos3_joint_video_hand_pose/docs/model_learning/overfit_v0.0_contract.md

## 11. 结果解释

### 情况 A：Video 提升，action alignment 提升

说明原双向 attention 中 future action -> future video 的耦合是重要问题。保留 video-first mask，再进入更大数据或更长训练。

### 情况 B：Video 提升，action 下降或不同步

说明保护 video 路径有效，但同步 sigma 下 action 只读取中间 noisy video 状态可能不够。下一轮单独比较：

- independent/video-leading action schedule；
- 或先生成 clean video、再做 inverse dynamics 的两阶段 sampler。

不要在 V1 直接混入这些改动。

### 情况 C：Video 不变，alignment 提升

mask 仍然有效：它改善了跨模态关系，但原视频问题可能主要来自数据规模、IT2V 适配或共享参数训练，而非前向 action 污染。

### 情况 D：Video 仍下降，alignment 无改善

先检查：

- mask 干预不变性是否真的成立；
- J0/J1 sample order 和 noise 是否一致；
- full_flex 控制组是否与 dense 等价；
- action condition/normalizer 是否正确；
- action loss 是否过度主导共享梯度。

只有结构测试全部通过后，才把“共享优化冲突”作为下一轮假设。即使如此，也应另做 stop-gradient/frozen-video/adapter ablation，不能修改本轮定义。

## 12. 验收标准

V1 被认为实现正确，必须同时满足：

1. dense mode 回归不变；
2. full_flex 与 dense 的 toy forward/backward allclose；
3. video-first mask truth table 完全正确；
4. 两层以上网络中，future action 扰动不改变 video prediction；
5. video 扰动能够影响 action prediction；
6. action loss 能更新共享 moe_gen；
7. training 与 inference 使用同一 mask；
8. 8 卡真实 packed batch 完成至少 3 steps；
9. J0-flex-full/J1-video-first 使用相同 checkpoint、样本顺序、noise 和超参数；
10. 100-step replay 后再决定是否进入 300-step。

## 13. 推荐命名

配置：

~~~text
egoverse_joint_video_hand_pose_v7_flex_full
egoverse_joint_video_hand_pose_v7_video_first
~~~

输出：

~~~text
outputs/joint_video_hand_pose/train/v7_flex_full/
outputs/joint_video_hand_pose/train/v7_video_first/

outputs/joint_video_hand_pose/inference/v7_flex_full/iter_*/
outputs/joint_video_hand_pose/inference/v7_video_first/iter_*/
~~~

第一轮不复用 overfit_v0.0 输出目录，也不覆盖当前 IT2V 训练产物。
