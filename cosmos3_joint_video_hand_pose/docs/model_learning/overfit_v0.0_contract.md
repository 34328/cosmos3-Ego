# EgoVerse × Cosmos 3：Joint Video–Action 模型/数据合同

> 文件名保留历史版本号；当前运行基线见
> [当前 Joint Overfit 基线](../training/current_joint_overfit_baseline.md)。
> 日期：2026-08-21  
> 状态：57D action、坐标、normalizer、visibility、loss 与 replay 的语义合同仍有效；
> 并行、token cap、noise schedule 和 LR 以当前运行基线为准。历史 V1–V6 以及
> CP2/85K 运行参数不再作为启动依据。
> overfit 数据：`brushing_shoes / repair_bench`，36 episodes / 181 segments。Mecka 100h train split 只用于冻结 codec/normalizer 统计，并作为后续正式训练母集。  
> 模型：从 `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464` 初始化，不修改 Cosmos Generator 主干结构。

> 讲解用总流程图与 action 子流程图见 [Data Pipeline](pipeline.md)；该图不替代本文的实现合同。

> 稳定语义为：MLP-AE-15、`lambda_out_of_fov=0`、`640x368`、`T=4n+1`、
> translation-scale action contract v2 与八个 action 子块等权。当前运行值为 75K、
> video/action `1.0:0.7`、CP1/FSDP8；这些运行 guardrail 不改变 57D 表示或坐标合同。

## 0. 当前结论

给定首帧 RGB、文本和任务提供的首帧双手 Pose，模型联合生成未来视频和未来双手 Pose：

```text
首帧 RGB + 文本 + 首帧真实双手 Pose
        ↓
Cosmos 3 原生 video/action Generator
        ↓
未来 RGB + 未来 camera Pose + 未来双手 Pose
```

手部 Pose 被视为 Cosmos 的 `action` modality，而不是另起一个 Pose 网络：

```text
Pose/camera state --(模型外 adapter)--> [T, 57] native action
                                      --(Cosmos尾部padding)--> [T, 64]
                                  ↓
                              action2llm
                                  ↓
                          Cosmos Generator
                                  ↓
                              llm2action
                                  ↓
Pose state <--(模型外 codec inverse)-- predicted action
```

Cosmos 的 clean/noisy 语义必须严格按源码理解：整段数据是加噪前的 clean ground truth `x0`；只有 `condition_mask=1` 的 action slot 保持 clean 并不计算 action flow loss，其余 slot 从同一份 `x0` 加噪后生成。

## 0.1 三个关键设计项的状态

<h3 style="color:#2e7d32">TODO-1（已完成）：Cosmos 原生 57D action adapter 与冻结 MLP-AE-15</h3>

> 原生 action 顺序固定为：`camera(9), right_wrist(9), right_fingertips(15), left_wrist(9), left_fingertips(15)`。camera/wrist 的 9D 是 `translation 3D + rotation rot6d 6D`；每只手的 20 个非 wrist keypoint（60D）使用该帧自己的 wrist-local 坐标。右手和左手始终使用各自独立的冻结 codec；codec 只处理手指局部形状，不处理 camera 或 wrist。原生 57D 送入 Cosmos 后由 `max_action_dim=64` 在尾部补 7 个 padding 通道。
>
> **方案 A：PCA-15。** 每侧使用冻结 PCA 将 `60D -> 15D`，inverse PCA 将 `15D -> 60D`。完整 test 的平均手尺度归一化误差为右手 `2.507%`、左手 `2.273%`，P95 为 `5.812%/5.273%`。
>
> **方案 B：冻结 MLP-AE-15。** Encoder 为 `60 -> 64 -> SiLU -> 32 -> SiLU -> 15`，Decoder 为 `15 -> 32 -> SiLU -> 64 -> SiLU -> 60`，每侧 13,003 个冻结参数。train 内部 5% episode holdout 仅用于调参，最佳配置为 `lr=5e-4, batch=8,192, epoch=120`；之后使用全部 train 重训。canonical seed-1 完整 test 的平均归一化误差为右手 `1.844%`、左手 `1.669%`，P95 为 `4.712%/4.405%`；独立 seed-2 复现为 `1.836%/1.585%`。方案 B 的 encoder/decoder 在 Cosmos 训练和推理中均冻结。
>
> **overfit_v0.0 锁定方案 B：`hand_codec.type=mlp15`。** 使用 canonical seed-1 的左右权重；PCA-15 只保留为后续 ablation，不允许在同一个 Cosmos checkpoint 中互换。首帧与未来、训练与推理、encode 与 decode 必须使用同一组冻结 MLP 权重及其 checkpoint 内的输入 mean/std、latent mean/std。Cosmos checkpoint/config 必须记录 codec 类型、左右权重 SHA256 和归一化版本。训练所需稳定权重与机器可读 manifest 位于 `cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v2_4/`。

<h3 style="color:#2e7d32">TODO-2（已实现并通过单元测试）：掌心 visibility 与逐手 action loss 权重</h3>

> TODO-2 的唯一详细规范见 [掌心 Visibility 与 Pose Loss v0.6](../specs/hand_visibility_loss_v0.6.md)。主方案不重复 loss 公式，避免两份文档再次漂移。  
> 已锁定：直接读取 `left/right.obs_ee_pose[:3]` 作为 GT 掌心，以 GT camera 和 episode 原始 intrinsics 投影到原始 `640x360 images.front_1` 画面；visibility 按 episode 逐帧离线计算并写回派生字段，训练 loader 只读取、不重算；不考虑画面内遮挡；首帧真实双手 Pose 始终作为 clean `a0`，不可见手不置零、不用 Null/sentinel；future 画外手默认按 `lambda_out_of_fov=0` 不计 pose loss，分子和分母使用同一 GT 权重；不新增 visibility 输出通道，原生 action 仍为 57D。`0.2` 和 `1.0` 仅作为后续对照实验。

<h3 style="color:#2e7d32">TODO-3（预测与 GT 可视化均已实现）：未来相机 Pose 与四格可视化</h3>

> camera Pose 沿用 Cosmos 原生 action 的 `camera(9)` 槽位：首帧是首帧相机初始状态（clean condition），未来帧是相对于 clip 首帧相机的变化量。原始 `640x360` 不 resize，只在底部 reflection-pad 8 px 得到 `640x368` VAE 输入；`image_size=[368,640,368,640]` 保证 VAE 不把 360 向下取整为 352，解码后才裁回顶部 `640x360`。episode 原始内参和投影坐标不变。正式 replay 顶部以黑底白字显示 JSON prompt 中的原始 segment 指令；主体为两行两列：第一行是生成视频上的预测双手投影和预测 `F0` 第三人称视图，第二行是原始 GT 视频上的 GT 双手投影和 GT `F0` 第三人称视图。具体见第 7 节。
>
> 早期 GT oracle 双视图曾独立验证投影几何。正式推理现已将该 GT 链路作为第二行，并与第一行预测结果使用同一 action decode、投影实现和第三人称空间尺度；历史验收目录不参与训练或正式推理运行。

<h3 style="color:#2e7d32">TODO-4（adapter/config 已实现，基础 smoke 已验收）：数据到 Cosmos 的视频训练链路</h3>

> 唯一详细规范见 [TODO-4：EgoVerse 视频训练链路 v1](../specs/todo4_video_training_pipeline_v1.md)。已锁定：原始 `640x360` 底部 pad 8 px 为 `640x368`；每个 sample 显式满足 `T=4n+1`；当前 75K exact-token cap；超限 segment 按原片段的 80/70/60/50% 分档同步均匀采样，50% 仍超限则丢弃；合法完整 samples 交给 Cosmos 原生 `PackingDataLoader` 动态 packing；所有视频、action、camera、pose、visibility 和 metadata 使用同一时间索引。

<h3 style="color:#2e7d32">TODO-5（已实现并通过单元测试）：Cosmos 原生 Flow Loss 与 Visibility Mask</h3>

> 视频生成 loss 完整复用 Cosmos 原生 flow-matching loss，不增加像素重建或 decoded-pose geometry loss。action loss 仍使用原生 flow-matching velocity MSE，只取前 `raw_action_dim=57`；`[57:64]` padding 永不参与 loss。camera translation/rotation、左右 wrist translation/rotation、左右 hand latent 共八个 active 子块等权；每只手的三个子块共享该侧 GT palm visibility mask，camera 两块始终有效。每个 sample 的分母只统计 active 子块，再走 Cosmos 原生跨 rank sample-level averaging。`lambda_out_of_fov=0`，当前 video/action 系数为 `1.0:0.7`。

<h3 style="color:#2e7d32">TODO-6（artifact 已完成，自动 checkpoint 绑定待决定）：最终 Normalizer 统计与冻结</h3>

> 已遍历 train split 并冻结 `cosmos3_action_contract/v2`。该统计来自原始训练分布，
> 不因 pack cap 从历史 85K 收紧到当前 75K 而重新拟合。state normalizer 保持 v1
> 字节一致；future camera/right-wrist/left-wrist translation 使用零中心、train-only
> std scale，18 个 rotation 通道保持 v1 q01/q99 参数不变；继续使用不 clamp 的可逆
> `PiecewiseAsinhNormalizer(beta=1)`。训练和 replay 启动脚本均先执行 v2
> manifest/hash validator。

TODO-1、TODO-2、TODO-3、TODO-4 和 TODO-5 已完成设计；TODO-6 已完成 artifact 统计与手动校验，但自动 checkpoint 绑定尚未接入。TODO-3 的预测/GT 四格 replay 已实现并通过真实推理样本验收；TODO-4 的 adapter/config 已实现并通过 8 卡三步基础 smoke。

## 1. 任务定义

训练窗口首帧条件为：

$$
c=\{I_0, P_0^L, P_0^R, y\},
$$

其中 `I0` 是首帧 RGB，`P0^L/P0^R` 是 Mecka 提供的首帧左右手真实 Pose，`y` 是 `Overall task: {task_description} Current segment: {text_normalized}`，再由 Cosmos 官方 action JSON formatter 封装。任一文本为空时退化为另一项。首帧掌心是否处于 RGB 有效画面内不改变 `a0` 的编码；visibility 只作为 future pose loss 的 GT metadata。

模型生成：

$$
\hat X=\{\hat I_{1:T-1},\hat C_{1:T-1},\hat P_{1:T-1}^L,\hat P_{1:T-1}^R\}.
$$

窗口共有 `T` 个物理 RGB 帧，索引为 `0...T-1`；首帧是条件，模型生成 `1...T-1`。本版本的数据合同始终提供双手 Pose，不设计缺失手 token。

这里的“联合”指未来 video latent 和未来 hand-pose action 在同一个 Cosmos Generator 中同时 flow matching，并在 attention 中互相读取；不是先生成视频、再从视频估计 Pose。

第一版明确不做：

- 不拟合 MANO `theta/beta`，不增加 mesh decoder；
- 不只预测指尖再用 IK 猜中间关节；
- 不另起 Pose DiT、Pose 双塔或 Pose 专用 velocity head；
- 不修改 Cosmos Generator 的主干 attention、MoT 和原生 action 投影结构。

第一版的参数训练边界固定为：Reasoner/LLM 理解塔冻结，视频 VAE tokenizer 冻结；共享 Generator 的 generation pathway 做全参数训练，不使用 LoRA-only 方案。训练参数包括共享生成主干 `moe_gen`、`time_embedder`、视频投影 `vae2llm/llm2vae`、原生 action 投影 `action2llm/llm2action` 和 `action_modality_embed`。这里的“Generator 全参数训练”是指 Cosmos 的生成路径全量更新，不是解冻 Reasoner/LLM 理解路径，也不是训练 VAE tokenizer。

## 2. Cosmos 3 原生 action 合同

### 2.1 三种模式的准确含义

Cosmos 源码的 mode 不是简单的“clean/noisy 二选一”：

| 模式 | Video | Action |
|:---|:---|:---|
| Forward Dynamics | future video 为生成目标 | action 全部 clean condition |
| Inverse Dynamics | video 为 clean condition | action 为 noisy 生成目标 |
| WAM/Policy | 首帧、initial state 或 history 为 clean condition；future video 和 future action 为生成目标 |

本项目采用源码中的 `wam` mode，并使用 `use_state=True` 等价的 Case B 时间布局：

```text
x0_tokens_action = [a0, a1, ..., aT-1]
condition_mask    = [ 1,  0, ...,    0]
```

- `a0`：首帧 camera、左右 wrist 和双手局部手型的真实初始状态，保持 clean，作为 action condition；无论掌心是否在 RGB 内都使用真实 Pose；
- `a1:T-1`：未来 camera/wrist 的首帧相对增量和每帧 wrist-local 手型，从同一份 clean `x0` 加噪并参与生成；逐时间、逐手 loss 权重使用 TODO-2 的 GT visibility 合同。这里 `T` 与 RGB 物理帧数相同。

Cosmos 源码中 `action2llm` 对 clean 和 noisy action 都使用；只有 noisy action 的 hidden state 通过 `llm2action` 预测 velocity。`condition_mask` 同时控制 sigma 和 action loss，不表示手是否在 RGB 内。

### 2.2 源码对应关系

| 行为 | 源码 |
|:---|:---|
| mode 生成 action condition indexes | `cosmos_framework/data/generator/action/transforms.py::build_sequence_plan_from_mode` |
| 构造 `[T,1]` action `condition_mask` | `cosmos_framework/data/generator/sequence_packing/sequence.py::pack_action_tokens` |
| `sigma_action *= (1-condition_mask)` 并生成 `xt_action` | `cosmos_framework/model/generator/omni_mot_model.py` action noising block |
| 所有 action token 经过 `action2llm` | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py::_encode_action` |
| 仅 noisy action 经 `llm2action` | `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py::_decode_action` |
| condition slot 不计 action flow loss | `cosmos_framework/model/generator/algorithm/loss/flow_matching.py` |

### 2.3 本项目不新增模型结构

本项目只新增数据 adapter 和模型外冻结的 15D hand-pose codec：

```text
hand/camera adapter -> [T,57] native action -> 尾部 pad 到 [T,64]
Cosmos Generator -> 原生 llm2action -> [T_noisy,64] action velocity
```

以下模块不属于 Cosmos 3 原生主路径，也不加入第一版：`PoseConditionEncoder`、`PoseActionAdapter`、graph attention、skeleton-aware MLP、Pose Output Adapter、独立 Pose velocity head。

## 3. 数据范围：Mecka 100h 母集与 overfit 子集

数据母集是 `/mnt/lzh/cosmos/training_manifests/` 下的 Mecka 100h manifests；清单中的 `abs_zarr_path` 继续指向 `/mnt/lzh/egoVerse/datasets/`。本文不引入其他数据来源、source embedding 或跨来源切分逻辑。codec 与 action normalizer 的冻结统计使用 100h train split。

当前 split：

```text
train episodes: 4142
test episodes: 153
val episodes: 0
train segments: 32355
test segments: 1180
```

当前 joint overfit 不直接训练全部 32,355 个 train segments，而是只使用冻结清单
`brushing_shoes_repair_bench_36ep_v1` 的 36 episodes / 181 train segments；
不另划 validation split，也不使用 test 数据。后续 100h 正式训练仍沿用同一模型、
adapter、normalizer 和 loss 合同，但训练规模与超参数另行冻结。

正式清单：

```text
mecka_100h_v1_episodes.csv
mecka_100h_v1_segments.csv
mecka_100h_v1_task_summary.csv
```

### 3.1 使用字段

每个 Zarr segment 需要读取：

| 字段 | 用途 |
|:---|:---|
| `images.front_1` | RGB 视频 |
| `left/right.obs_wrist_pose` | wrist translation + quaternion |
| `left/right.obs_ee_pose` | `[:3]` 是 SLAM-world 掌心中心，仅用于 TODO-2 GT visibility 投影；不进入 action，也不增加模型输入维度 |
| `left/right.obs_palm_in_fov_front_1` | 已离线物化的逐帧 `uint8 [total_frames]`；按原始 `640x360 images.front_1` 与原始 episode intrinsics 计算，loader 直接读取为 `visibility_gt` |
| `left/right.obs_keypoints` | 21 个手部空间关键点 |
| `obs_head_pose[0:T]` | 构造 action 的 `camera(9)`，并用于世界坐标到首帧参考系的变换和 TODO-3 投影验收 |
| root attrs `intrinsics["front_1"]` | 每个 episode 的相机投影矩阵 `[3,4]`；实际读取为 `group.attrs["intrinsics"]["front_1"]`，用于 TODO-2 可见性与 TODO-3 投影，不是逐帧 Zarr array |
| segment `start_idx/end_idx` | 子任务窗口边界 |
| segment caption / task description | 文本条件 |

不得使用底层数组 shape 替代 Zarr 的 `total_frames`。窗口严格按 manifest 的 segment 起止帧读取，避免把无关帧和错误 caption 混入样本。

### 3.2 Caption 规则

overfit_v0.0 固定使用 `prompt_mode="episode_context_and_segment"`：当两项都存在时，先构造 `Overall task: {task_description} Current segment: {text_normalized}`；任一项为空时使用另一项。随后使用 Cosmos 官方 action JSON，将该完整文本放入 `actions[0].description`，`viewpoint="ego_view"` 由 formatter 写入 `cinematography.framing`，duration/time 使用浮点秒。`hand_pose/domain_id=3` 是独立 action domain，不是文本 tag。sampler 和 packer 必须按最终 JSON 字符串的实际 token 数计预算。不使用根层无意义的 `debug` 作为训练文本。首帧必须位于对应 segment 的时间范围内。

### 3.3 TODO-4：数据与视频训练接口

详细实现合同只维护在 [TODO-4：EgoVerse 视频训练链路 v1](../specs/todo4_video_training_pipeline_v1.md)。简要结论：adapter 读取完整 manifest segment，显式对齐到 `T=4n+1`，超 75K 时按 80/70/60/50% 分档保留首帧并对 future 同步均匀采样；Cosmos 原生 packer 只负责动态组合已经合法且未超 cap 的 samples，不负责切分或修复单个 sample。

#### 3.3.1 Domain-3 action projection 加载合同（P0 已锁定）

`/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464/model` 的 DCP metadata 与定向 tensor 审计已经确认：该 checkpoint 使用 `max_action_dim=64`、`num_embodiment_domains=32`、hidden size 4096，并完整保存 regular/EMA 两套 action projection：

```text
action2llm.fc.weight    [32, 64*4096]
action2llm.bias.weight  [32, 4096]
llm2action.fc.weight    [32, 4096*64]
llm2action.bias.weight  [32, 64]
action_modality_embed   [4096]
```

初始化为零的两组 bias 中，实际非零 domain 行恰好为：

```text
[1,2,3,6,7,8,12,13,15,20]
```

它们与源码登记的 AV、camera pose、human hand pose、UMI、Bridge、DROID、双臂/UR、AgiBot 和 Fractal domain 对应。domain 3 的 `action2llm/llm2action` bias L2 分别为 `1.2625/0.2010`，regular 与 EMA 都非零，因此 `hand_pose -> domain_id=3` 是真实训练过的 checkpoint 行，不是未训练占位。审计脚本与报告为：

```text
/mnt/lzh/egoVerse/tmp/cosmos3_domain3_checkpoint_audit_20260814/audit_domain3_checkpoint.py
/mnt/lzh/egoVerse/tmp/cosmos3_domain3_checkpoint_audit_20260814/report.json
```

该 checkpoint 的原生 hand-pose 57D 语义是：

```text
[camera(9), right_wrist(9), right_fingertips_xyz(15),
 left_wrist(9), left_fingertips_xyz(15)]
```

本方案保持同一 57D 槽位顺序，但两个 15D 块使用冻结 MLP-AE-15 latent。尽管其数值语义与原始 fingertip xyz 不同，第一版仍按用户决定完整继承 Cosmos 的预训练 action projection，不重置任何参数或切片。加载合同为：

- `domain_id=3`，`num_embodiment_domains=32`；
- 从 base checkpoint 加载 regular `action2llm`、`llm2action`、它们的 bias 和共享 `action_modality_embed`；当前配置 `keys_to_skip_loading=["net_ema."]` 且 `ema.enabled=false`，不创建新 EMA，训练与推理均使用直接优化的 `net.*`；
- 上述参数不放入 `keys_to_skip_loading`；
- checkpoint 加载后不执行 slice reset、bias 清零、重新初始化或其他 post-load 修改；
- 对 checkpoint path、shape、missing/unexpected keys 和加载成功状态做 fail-fast 断言；
- 后续训练可按方案更新 Generator 参数，但“从 base checkpoint 加载”这一步不人为破坏已有预训练权重。

#### 3.3.2 EgoVerse canonical `hand_pose` raw width（P0 已锁定）

Cosmos 的 `ActionProcessor` 从输入 action 的最后一维自动记录 raw width。EgoVerse 遵循官方 hand-pose layout：

```text
camera(9) + right_wrist(9) + right_fingertips(15)
+ left_wrist(9) + left_fingertips(15) = 57D
```

因此训练侧不在 `domain_utils.py` 添加全局 `hand_pose -> 57`；由 `ActionProcessor` 根据 `[T,57]` 自动产生 `raw_action_dim=57`，再按原生流程 pad 到 `[T,64]`。

通用 inference helper 采用显式 override：EgoVerse 配置传入 `raw_action_dim=57`，helper 使用显式值而不查找全局 `domain_utils.py` 默认表。该值只用于输入 action 的 unpad、输出 action 的 truncate 和回录尺寸校验；不参与 action 数值计算，不改变 padding 之前的 57D、不改变 64D 模型输入、noise/flow target、loss mask、sampler 或任何模型权重。

验收只需断言：`[T,57] -> pad [T,64] -> inference unpad [T,57]` 后 shape 不变，数值与原始 57D action 一致；不需要重新训练或修改 checkpoint。

### 3.4 视频/VAE 接口

Mecka `images.front_1` 原始 shape 为 `360x640x3`。不 resize；仅在底部 reflection-pad 8 px，形成 `[3,T,368,640]` 模型输入。Wan2.2 VAE 明确要求 `T=1` 或 `(T-1)%4=0`，adapter 必须在进入 VAE 前保证该条件。生产预处理必须绕开官方 `VideoResize` bucket snapping，避免被改成 `832x480`。TODO-2 visibility 仍以原始 `640x360` 有效域为准。

## 4. 手部 Pose 表示

### 4.1 已确定的 camera/wrist 时间语义

<h3 style="color:#c62828">已确定合同：首帧是初始状态，未来是相对首帧的增量</h3>

所有 camera/wrist Pose 都先表达在首帧相机坐标系中：

```text
t=0:
  camera 9D      = 首帧相机初始状态（参考原点）
  right/left wrist 9D = 首帧左右腕在首帧相机坐标系中的位置和旋转

t=1...T-1:
  camera 9D      = 相对于首帧相机的变化量
  right/left wrist 9D = 相对于首帧对应 wrist 的变化量
```

这里的“相对”是相对于 clip 首帧，不是 `t-1 -> t` 的逐帧差分。该布局对应 Cosmos `use_state=True` 的 `a0` clean initial state 与 future action targets。

### 4.1.1 锚定增量与 mRoPE 的关系

mRoPE 只负责把 `action[t]` 对齐到第 `t` 个物理时刻，不规定 action token 内必须存绝对状态、逐帧差分或相对首帧增量。Cosmos `pose_utils.py` 同时实现了 `backward_framewise` 和 `backward_anchored`，因此当前首帧锚定增量与 action 的绝对时间位置不冲突。不能仅因为 action token 具有等间隔时间位置，就推导出必须改成 `t-1 -> t` 逐帧差分。

当前选择仍是首帧锚定增量：它避免逐帧积分漂移，并允许每个 future token 直接恢复相对于首帧的状态。代价是 camera/wrist 增量分布会随预测时刻变宽，必须由下面的归一化合同处理。

### 4.2 已确定的手指局部状态

<h3 style="color:#c62828">已确定合同：每一帧都使用该帧自己的 wrist-local 坐标</h3>

对第 `j=1...20` 个非 wrist keypoint，每一帧都以该帧自己的 wrist 为原点、使用该帧自己的 wrist rotation：

$$
q_{t,j}^h=(R_t^h)^\top(p_{t,j}^{world}-t_t^h).
$$

每手的原始手指形状为 `20 × 3 = 60D`，经该侧冻结的 MLP-AE-15 encoder 压缩为 Cosmos 原生 `15D` fingertip 槽位。首帧和未来帧使用同一个 codec，因为二者都是同一语义的当前帧 wrist-local 手型；手指点不再减去首帧手型。

对应的原生 action 结构为：

```text
camera                   9D
right wrist              9D
right hand MLP latent   15D
left wrist               9D
left hand MLP latent    15D
--------------------------------
native action           57D
Cosmos padded action    64D
```

MLP-AE codec 只使用左右 checkpoint 内各自的 input mean/std 与 latent mean/std 做归一化和反归一化；当前实现不使用首帧掌尺度或骨长模板，也不预测 MANO `beta`。

### 4.3 TODO-1 的 MLP-AE-15 验收结果

TODO-1 已完成。最终超参数先在 train 内部 5% episode holdout 上选择，选择过程不读取原始 test；随后 canonical seed-1 在全部 4,142 个 train episodes 上重训，每侧使用 1,060,352 个有效训练帧，并在全部 153 个 test episodes、每侧 388,424 帧上验收。codec 只处理每手 `60D -> 15D -> 60D`，camera/wrist 不经过 codec；encoder/decoder 在 Cosmos 训练和推理中始终冻结。

| 侧别 | Test 帧数 | 平均点误差 | 平均手尺度归一化误差 | P95 归一化误差 |
|:---|---:|---:|---:|---:|
| Right | 388,424 | 2.083 mm | 1.844% | 4.712% |
| Left | 388,424 | 1.979 mm | 1.669% | 4.405% |

overfit_v0.0 canonical 冻结文件：

```text
right_mlp15_primary.pt  SHA256 1173b50ef35a1d0eab5b36e2c4ba5a0e0a34c7400cbc3043387669e2cfd1f0c3
left_mlp15_primary.pt   SHA256 3e11c7e535a968a08043b7e281b909e7689ed87708eb999126979de6b2be53a2
```

独立 seed-2 复现的平均归一化误差为右手 `1.836%`、左手 `1.585%`，用于证明训练稳定性，不替代 canonical seed-1 权重。历史完整报告保留在 `/mnt/lzh/egoVerse/tmp/cosmos3_pose_representation_validation_20260813/codec15_pca_linear_mlp/tuned_nonlinear_full_test_report.json`，不参与训练运行。

### 4.4 camera/wrist action 归一化合同

<h3 style="color:#c62828">已确定合同：a0 initial state 与 future anchored delta 分开归一化，禁止硬裁剪长尾</h3>

当前 action 的 camera/wrist 槽位在时间上具有两种语义：

```text
a0:       initial absolute state
a1:T-1:  relative-to-first-frame anchored delta
```

两者虽然占用同一原生字段，但不能用一套 future-delta quantile 统计直接归一化。对 21,384 个 train 4 秒窗口的审计显示：若用 future delta 的 q01/q99 归一化 a0，右腕有 `98.46%`、左腕有 `99.70%` 的 a0 token 至少一个通道越出该范围。

future anchored delta 也存在明确的时间相关长尾。4 秒末的分布为：

| 字段 | Translation P50 / P95 / P99 | Rotation P50 / P95 / P99 |
|:---|:---|:---|
| Camera | 0.021 / 0.290 / 0.758 m | 2.86 / 22.78 / 53.03 deg |
| Right wrist | 0.059 / 0.366 / 0.843 m | 28.37 / 85.68 / 115.47 deg |
| Left wrist | 0.044 / 0.341 / 0.801 m | 23.39 / 86.81 / 121.87 deg |

因此固定以下实现合同：

1. `a0` 使用仅由 train split initial-state token 统计得到的 `state_normalizer`；
2. `a1:T-1` 使用仅由 train split future anchored-delta token 统计得到的 `delta_normalizer`；
3. future delta 不使用 `clamp(-1,1)`。优先采用 Cosmos 已实现的可逆 `piecewise_asinh_rot`，在 q01/q99 主区间保持线性，对长尾做可逆压缩；若实现选择其他 normalizer，必须同样无损可逆；
4. MLP 15D hand latent 的首帧和未来语义相同，使用同一套 canonical seed-1 冻结 codec；其输入 mean/std 与 latent mean/std 来自权重文件，独立于 camera/wrist 的 state/delta normalizer；
5. 归一化验收必须按 `t=1,15,30,60,90,120` 分别报告 normalized P50/P95/P99/max、越界率和 inverse round-trip error，不能只报告混合全时序统计；
6. `state_normalizer` 与 `delta_normalizer` 的统计文件、训练 split、字段顺序、方法、参数和 SHA256 必须写入 checkpoint/config 元数据。

Cosmos 原生 `human_hand_pose_lerobot_dataset.py` 在 quantile normalization 后执行 `result["action"].clamp(-1.0, 1.0)`。该行为适用于它当前的 16-step framewise action，但不能原样用于本项目 120-step anchored delta，否则晚时刻的有效大位移/旋转会被截断且不可逆。本项目 adapter 必须绕开这项 clamp，并在 inverse decode 前按 token 类型选择对应的反归一化器。该改动只属于数据/action preprocessing，不修改 Cosmos Generator 架构。

旧 121 帧审计报告已归档为 `cosmos3_joint_video_hand_pose/artifacts/cosmos3_action_contract/v1/reports/legacy_121_frame_report.json`，仅供分析。

当前机器可读 action artifact 绑定包位于 `cosmos3_joint_video_hand_pose/artifacts/cosmos3_action_contract/v2/`；v1 仅作为 v2 的不可变来源和历史对照保留：

- `manifest.json` 固定 57D/64D 字段顺序、时间语义、canonical seed-1 左右 codec 路径与 SHA256、train manifests SHA256 和 checkpoint 绑定规则；
- 现有 codec 部分已标记为 `ready`；
- state/future-delta normalizer 已按 TODO-4 实际 train samples 生成并绑定 SHA256；
- 旧 121 帧审计报告只登记为 `analysis_only_not_a_training_artifact`，禁止当作最终 normalizer 加载；
- `validate_manifest.py` 当前校验 state/future normalizer、normalizer report 的 SHA256 和 translation-scale 合同；overfit_v0.0 的训练与 replay 脚本启动时自动调用，失败则不进入模型加载。codec/source-manifest 全量哈希与 checkpoint metadata 的自动绑定仍属于 TODO-6 未接入部分，不能宣称已经自动验收。

## 5. 联合生成与训练

### 5.1 数据到 Cosmos action

```python
native_action_57 = build_native_action_sequence(...)  # [B, T, 57]
normalized_action_57 = normalize_by_slot(
    initial=native_action_57[:, :1],       # state_normalizer
    future=native_action_57[:, 1:],        # delta_normalizer for camera/wrist
)
x0_tokens_action = pad_action_to_max_dim(normalized_action_57, 64)  # [B, T, 64]
condition_mask_action = [1, 0, ..., 0]
```

视频端：首帧 RGB 是 clean visual condition，future RGB latent 是 noisy video target。

Action 端：`x0_tokens_action[0]` 是 clean initial-state condition，`x0_tokens_action[1:]` 是从 clean ground truth 加噪后的 future hand action。

### 5.2 Flow matching

当前稳定 baseline 使用独立模态 noise schedule（`independent_action_schedule=True`）：

$$
x_{v,\sigma_v}=\sigma_v\epsilon_v+(1-\sigma_v)z_v,
$$

$$
x_{a,\sigma_a}=\sigma_a\epsilon_a+(1-\sigma_a)a.
$$

同一个 sample 的 video/action 分别采样 `sigma_v` 与 `sigma_a`；各模态再用自己的
condition mask 将 clean slot 的 sigma 置零。独立 schedule 是当前 baseline 的固定
组成，不是 Video-First mask 额外改变的变量。Cosmos 对 action 使用：

```text
sigma_action[t] = sampled_sigma_action * (1 - condition_mask_action[t])
```

因此首帧 action 不加噪；future action 加噪。所有 action token 走 `action2llm`；只有 noisy action 走 `llm2action` 输出 `v_action`。

### 5.3 TODO-5（已实现并通过单元测试）：Loss 合同

1. **Video loss**：完整复用 Cosmos 原生 flow-matching velocity MSE；condition video token 不计 loss；不增加 RGB 像素 loss 或 decoded-pose geometry loss。
2. **Action loss**：完整复用 Cosmos 原生 flow-matching velocity MSE，但在计算前截取 `raw_action_dim=57`，固定排除 `[57:64]` padding。
3. **Action 分组**：camera `[0:9]` 单独计算；right group `[9:33]`、left group `[33:57]` 分别包含 wrist 9D 和 hand latent 15D。
4. **Visibility mask**：future action 的 camera loss 始终计算；每侧 hand group 使用 GT palm visibility 权重。overfit_v0.0 默认 `lambda_out_of_fov=0`，画外侧不计该侧 pose loss；有效 group 的分子和分母同步重算，不能因屏蔽画外帧而人为降低 loss 尺度。首帧 `a0` 为 clean condition，不计 action flow loss。
5. **Action 子块**：camera translation/rotation、左右 wrist translation/rotation、左右 hand latent 共八块等权；某侧在该 segment 完全无有效 future frame时跳过该侧三个子块，分母按实际 active 子块重算。
6. **总权重与跨 rank 归一化**：`loss_scale=1.0`、`action_loss_weight=0.7`。rank 内对 segments 等权平均，并启用 `sample_level_loss_averaging=True`，由 Cosmos 按各 rank 的本地 sample 数校正整个 video+action loss，得到跨 rank 全局 sample mean。
7. **时间权重**：visibility/noisy mask 同时进入分子和分母；Cosmos `train_time_weight` 只乘分子，严格保持官方 flow-matching 语义。当前 `train_time_weight="uniform"`，数值恒为 1。

当前实现只在 Cosmos 原生 action flow loss 外增加逐时间、逐 hand group 的 GT 权重；Generator 架构和 video loss 保持不变。

## 6. TODO-2（已实现并通过单元测试）：掌心 visibility

本节只保留主方案需要的接口结论。所有 loss 公式、边界情况和单元测试要求以独立文档 [掌心 Visibility 与 Pose Loss v0.6](../specs/hand_visibility_loss_v0.6.md) 为唯一依据。

```text
palm_world[t, side] = left/right.obs_ee_pose[t, :3]
palm_cam            = inverse(obs_head_pose[t]) @ palm_world
visibility_gt       = palm_cam.z > 0
                      and projection(P_source, palm_cam) inside 640x360
```

- `obs_ee_pose[:3]` 是 SLAM-world 掌心中心；不需要每个样本再由 21 点重算。21 点平均只用于离线一致性审计；
- 上述结果已按 episode 全部 `total_frames` 离线写入 `left/right.obs_palm_in_fov_front_1`；训练窗口只按同一个 `[window_start,window_end)` 切片，不在 loader 中重复投影；
- 不考虑画面内遮挡，掌心在有效 RGB 画面内即 `visibility_gt=1`；
- 首帧真实左右手 Pose 始终进入 clean `a0`，不根据 visibility 置零或标记 missing；
- future 画外手仍保留真实 pose target，默认相对权重为 `0`；视频和 camera 不受其影响；
- `visibility_gt` 来自 GT 并停止梯度，不增加模型输出，`[57:64]` 继续只做 padding；
- 推理后可由预测 camera、预测 21 点与 intrinsics 确定性派生 `hand_in_fov` metadata。

全量 10,700,531 个 manifest segment 帧中，按原始 `640x360` 图像域计算的左/右掌心在画面内比例分别为 `99.9137%` 和 `99.8657%`，画外分别为 9,231 和 14,369 帧。实现只使用原始域离线字段；该统计不能再与旧的“21 点全部出界”统计混用。

2026-08-14 已完成 4,295/4,295 个 episode 的物化和独立只读审计。两个字段均为 `uint8 [total_frames]`、只含 0/1、array attrs `complete=true`，且 Zarr 根 `features/derived_annotations` metadata 完整。独立审计报告为 `/mnt/lzh/egoVerse/tmp/cosmos3_palm_visibility_fields_20260814/audit_report.json`；它重新读取所有数组后得到 manifest 画外计数 `left=9,231/right=14,369`。该报告不参与训练运行。

## 7. TODO-3（预测与 GT 可视化均已实现）：未来相机 Pose 与视频投影对齐

<h3 style="color:#c62828">已确定合同：camera Pose 使用原生 action 的 camera(9) 槽位输出</h3>

> 模型随 future action 一起预测相对于 clip 首帧的 camera Pose 增量，不新增 camera encoder、视频 token 占位或其他输出分支。几何目标仍是：
>
> $$
> p_t^{C_t}=T_{C_t\leftarrow F_0}p_t^{F_0},\qquad
> uv_t=\Pi(K_t,p_t^{C_t}).
> $$
>
> 视频不 resize，只在底部补 8 px 到 `640x368`，因此当前 episode 的原始内参 `K`、投影坐标和原始 `640x360` 有效域均保持不变。预测 action 和 GT action 都必须先经过同一个 inverse normalizer、rotation 6D 恢复、首帧相对 Pose 恢复和 MLP-AE-15 decode，得到 `F0` 中的 headcam 与 21 点双手，再做投影或第三人称绘制。正式 replay 顶部显示原始 segment 指令；第一行显示预测的生成视频投影与 `F0` 视图，第二行显示 GT 原视频投影与 `F0` 视图。预测和 GT 的两个第三人称面板必须共享空间范围和比例尺，四个面板使用同一帧索引。外部视觉里程计只能作为辅助诊断，不能替代模型的 camera action 输出。

## 8. 训练与实现验证

本项目不采用 Stage 1/2/3 分阶段训练；直接从 Nano SFT checkpoint 做一次 joint video + action fine-tuning。实现前后完成以下检查：

- 随机抽样 Mecka train 数据，验证 world -> `F0` -> wrist-local -> inverse 的闭环；
- 验证 `Delta W(0->0)=I`、quaternion 顺序和 rotation 6D；
- 验证 RGB、segment start/end、caption 和 Pose 使用同一帧索引；
- 完成 TODO-1 的 codec 重建报告；
- 验收 TODO-2 的掌心投影 overlay、全量统计和 loss 单元测试；
- 验收 TODO-4 的 `4n+1`、75K cap、五档同步索引、单长样本和多短样本 packing；
- clean：首帧 RGB、文本、`a0`；
- noisy：future video latent、future hand action `a1:T-1`；
- 复用原生 `action2llm`、`llm2action`、joint flow-matching 和 `PackingDataLoader`；
- `action2llm/llm2action` 与共享 Generator generation pathway 全参数共同训练，Reasoner 和 VAE 继续冻结；
- 不新增 Pose encoder/head。

## 9. 数据与模型接口

Loader 和 batch 必须遵循 TODO-4：视频 `[3,T,368,640]`、raw action `[T,57]`、Cosmos padded action `[T,64]`、`domain_id=3`、`mode="wam"`；slot 0 clean、future noisy。caption、condition mask、sequence plan、sample boundary 和所有模态使用采样后的同一 `T`。

## 10. 推理与评估

### 10.1 推理

1. 输入首帧 RGB、caption 和任务提供的首帧真实双手 Pose；
2. 首帧 camera/wrist/hand Pose 经 native adapter 得到 clean `a0`；
3. 初始化 future video/action noise；
4. 使用 Cosmos sampler 迭代更新 video 和 future action；
5. native action adapter inverse 得到 future camera、左右 wrist 和双手 20 点；
6. TODO-3 按原始 episode 内参逐帧投影，并按 TODO-4 采样后的时间索引输出顶部指令和预测/GT 两行两列 replay。

### 10.2 Pose 正确性

- 3D：wrist translation、rotation geodesic、local MPJPE、recovered trajectory MPJPE、bone-length drift、velocity、jitter；
- Codec：train-only fit，test reconstruction error；
- 几何：原始内参投影 overlay 与 `F0` 第三人称双手/headcam 轨迹。

### 10.3 视频与 Pose 一致性

当前版本已经从原生 action 输出 camera Pose。生成视频的投影对齐指标暂不锁定，可使用以下非几何辅助诊断：

- 生成视频的外部 2D hand keypoint/hand mask estimator；
- 视频动作事件与 Pose velocity/事件时间的一致性；
- 左右手活跃趋势、手型和手—物交互的人工检查。

严格逐帧 3D-to-generated-video 投影使用当前 episode 原始内参；是否进一步引入自动量化指标可在 smoke test 后讨论。

### 10.4 训练期 checkpoint 与人工监控

overfit_v0.0 不建立独立 val，不计算 test flow-matching loss，也不以数值 test loss 自动选择 checkpoint。test flow loss 在技术上可以通过对 GT 加噪后计算 velocity error 得到，但它依赖采样的 noise/time，并不能直接代表自由采样视频的主观质量；本阶段只做人工生成检查。

overfit_v0.0 的 36-episode 单任务实验在每个原生 checkpoint 保存点执行监控。达到保存点后先确认 DCP 完整落盘，再释放训练 ranks 并使用独立推理进程生成 replay：

```text
保存完整可恢复 checkpoint（300/600/900/1200/1500/1800/2000）
释放训练 ranks 后，对固定四个较长 train segments 做生成
保存 sample id、prompt、seed、checkpoint step 和全部可视化产物；各 checkpoint replay 分目录保存
```

四个 monitor samples 从该 36-episode train subset 中按固定 seed `42` 选择，来自不同 episode，帧数限制为 `201...401` 且原生满足 `T=4n+1`。推理使用各自完整 segment 的原始帧数，不人为统一长度。样本列表和 sampler seed 必须保存，以便所有 checkpoint 使用完全相同的输入。当前 overfit 不生成 test replay，也不使用 test 自动选择 checkpoint。

每个样本至少导出：

1. 原始生成视频；
2. 预测 action、GT action、首帧条件和原始 GT 视频；
3. H.264 四格 replay：顶部显示原始 segment prompt，第一行显示 prediction，第二行显示 GT；每行左侧为视频投影，右侧为 `F0` 双手/headcam 第三人称视图，预测与 GT 共享第三人称空间尺度。

## 11. 主要风险与检查顺序

1. Pose loss 不下降：先检查 codec normalization、quaternion 顺序、`condition_mask=[1,0,...]`、`action2llm` 输入和 action loss mask。
2. 视频变差：先检查 video/action raw loss、学习率、生成时 shift 与 prompt；权重或可训练范围的任何调整必须作为新实验版本，不能静默改写本合同。
3. Pose 与视频不同步：检查 RGB/action 是否使用同一 temporal index，并区分“预测 camera action 错误”和“生成视频视角与预测 camera 不一致”。
4. 骨长漂移：检查 codec 重建、wrist-local encode/decode、首帧恢复和异常数据；当前合同没有 bone loss。
5. 左右手交换：原生左右手块固定为右手 `[9:33]`、左手 `[33:57]`；若做镜像增强，必须同时交换 wrist、codec latent、visibility metadata 和左右手标签，camera `[0:9]` 不交换。

## 12. MVP 配置与最终合同

| 项目 | 值 |
|:---|:---|
| Base checkpoint | `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464` |
| overfit 数据 | `brushing_shoes / repair_bench`；36 episodes / 181 train segments；不另划 val、不使用 test |
| 统计母集 | Mecka 100h train split，用于 canonical codec 与 action normalizer；后续正式训练沿用同一数据/模型合同 |
| 当前 Zarr 训练帧 | 640x360（横屏，W x H；数组 shape 为 360x640x3） |
| RGB/Pose | 30 FPS / 30 Hz |
| 视频输入/VAE | 原始 640x360 不 resize；底部 reflection-pad 8 px 到 640x368；`T=4n+1` |
| Window / packing | 完整 manifest segment；75K cap；超限按 80/70/60/50% 同步均匀采样，仍超限则丢弃；Cosmos 原生 dynamic packing |
| Action width | 原生 57D，Cosmos 尾部 padding 到固定 64D |
| Action condition | slot 0 clean，`condition_mask=[1,0,...,0]` |
| Pose input/output | 模型外 native adapter；双手 20 点固定使用 canonical seed-1 `mlp15`；PCA-15 仅作后续 ablation |
| Generator | Cosmos 原生 action2llm、MoT、llm2action |
| Trainable parameters | `action2llm/llm2action` + 共享 Generator generation pathway 全参数；Reasoner 与 VAE 冻结 |
| Camera Pose | 原生 action 的 `camera(9)`；首帧初始状态，未来相对首帧增量 |
| Visibility | TODO-2 已实现；GT 掌心投影，默认画外权重 0，不增加模型输出通道 |

具体时间与 batch 合同见独立 TODO-4 规范。

## 13. 方案结论

第一版严格复用 Cosmos 3 的 action 机制：

1. Mecka segment 首帧 RGB、caption 和真实首帧双手 Pose 作为条件，不用 visibility 改写 `a0`；
2. native adapter 按 Cosmos 原生顺序构造 57D action；每手 60D wrist-local 手型经 canonical seed-1 冻结 MLP-AE-15，Cosmos 再尾部 pad 到 64D；
3. slot 0 是 clean initial-state condition，future slots 从 clean `x0` 加噪；
4. clean/noisy action 共用 `action2llm`，future noisy action 由 `llm2action` 预测 velocity；
5. video 和 future action 在同一个 Cosmos Generator 中联合 flow matching；
6. 模型外 native adapter inverse 输出 camera、左右 wrist 和完整 3D 双手 Pose；
7. TODO-1 至 TODO-6 的训练侧设计均已完成；TODO-2/4/5/6 已有对应实现或 artifact，TODO-3 的 prediction/GT 四格 replay 已实现并通过真实样本验收；TODO-4 adapter/config 已通过 8 卡三步基础 smoke；
8. EgoVerse adapter、训练配置和 smoke 入口已在 Cosmos 核心源码之外实现；不新增或修改 Cosmos Generator 模型结构。

## 14. 正式训练前仍需补齐的非-loss事项

TODO-2/5 已完成实现与单元测试，TODO-4 基础 smoke 已通过，但下面项目仍未全部达到“可以直接跑 100h 正式训练”的程度。它们不要求修改 Cosmos Generator 主干，却会直接影响实验是否可复现；应在正式训练前逐项完成。

| 优先级 | 项目 | 当前缺口 | 建议的冻结标准 |
|:---:|:---|:---|:---|
| P0 已锁定 | Overfit 数据与监控协议 | 当前 joint overfit 使用 brushing-shoes 36 episodes / 181 segments | 每 300 step 保存可恢复 checkpoint；至少运行并验收到 1200 step，是否延长到 2000 由曲线和 replay 决定；不计算 test loss |
| P0 已锁定 | Hand codec | 主 run 使用冻结 MLP-AE-15，PCA-15 仅作后续 ablation | 固定 `hand_codec.type=mlp15` 和 canonical seed-1 左右权重 SHA256；checkpoint 同时绑定输入/latent normalization；训练与推理不允许替换 codec |
| TODO-4（75K 已进入长程） | Window sampler / temporal packing / VAE / batch schema | `640x368`、`4n+1`、75K cap、80/70/60/50% 同步超长采样和原生 dynamic packing 已锁定；历史 CP2/90K 在第 39 步 OOM | 保持 CP1/FSDP8 与 expandable allocator；扩展 100h 前重新做 75K 全量审计 |
| TODO-5（已实现并通过单元测试） | Loss 函数与权重 | Cosmos 原生 video loss 不变；action 只取 57D，八个 active 子块等权并跨 rank 做全局 sample mean | 当前使用 `loss_scale=1.0`、`action_loss_weight=0.7`、`lambda_out_of_fov=0`；padding/condition/visibility/sample 等权测试必须持续通过；不新增额外 loss |
| P0 已锁定；checkpoint audit 已通过 | Domain/action projection compatibility | domain 3 是真实预训练行；本项目只复用 57D 宽度、槽位和 projection，MLP latent/anchored delta 并非原始 action 数值语义 | 按 3.3.1 节加载 regular `action2llm/llm2action` 及其 bias、`action_modality_embed`；不重置；跳过 base EMA，关闭新 EMA，直接保存和推理 `net.*` |
| P0 已锁定 | EgoVerse canonical `hand_pose` raw width | 训练侧 `ActionProcessor` 会从官方 57D action shape 自动记录 `raw_action_dim=57`；缺口仅在通用 inverse/WAM inference helper 不会从全局 `domain_utils.py` 推断本项目的 canonical 宽度 | 保持全局 `domain_utils.py` 不变；EgoVerse inference 入口显式传入 `raw_action_dim=57`，并让 helper 遵循“显式 override 优先于 domain 默认表”。该 override 只用于 unpad/round-trip 的尺寸元数据，不改变 action 数值、padding、noise、flow target、loss 或模型权重；用一次 inference round-trip 验收 |
| TODO-6（normalizer 已冻结；checkpoint 自动绑定未接入） | Final normalizer artifacts | 全部 32,355 个 train samples 已按 TODO-4 统计；state/future-delta 文件、方法、参数和 SHA256 已冻结。当前启动 validator 自动校验两个 normalizer、report hash 和 translation-scale 合同，但尚未把完整 artifact metadata 写入 checkpoint | 自动绑定接入后，训练、resume 和 inference 必须核对 artifact manifest、codec、两个 normalizer 和 source manifests 的 SHA256；旧 121 帧报告不得复用 |
| ⚠️ 实现阶段 Warning | Base checkpoint/tokenizer smoke test | 当前不是已发现的 checkpoint 问题，仅用于防止实现配置误加载或误训练 | 加载 `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/` 后确认 Reasoner/LLM 与 video VAE 冻结，保留并加载预训练 generation/action 参数；使用一批数据成功完成 forward 和 backward，并确认冻结模块无梯度、待训练模块有有效梯度即可 |
| 已锁定 | Single-stage checkpoint inheritance | 不采用 Stage 1/2/3 分阶段训练 | 直接从 `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/` 启动一次 joint video + hand-pose fine-tuning；完整继承预训练 generation/action 参数，不重置 domain-3 action projection；Reasoner/LLM 和 video VAE 冻结 |
| P0 已锁定 | overfit optimizer/training schedule | 8 卡 `CP=1/FSDP=8`、BF16、full activation checkpoint、base LR `2e-5`、shared/video `4x`、action projection `5x`、warmup 100、cosine 2000 steps、grad accumulation 1、EMA off | 运行配置以当前 CP1 TOML 为准；每 300 step 保存完整 DCP 与 dataloader 状态；正式 run 持续记录 raw non-finite 与 grad clip |
| P1（暂缓） | Data quality audit | 数值有效性、异常跳变和坏帧尚未做完整批量审计 | 正式 100h 训练前做最低限度的 NaN/Inf、长度、Pose jump 和 RGB decode 检查；当前不展开复杂过滤规则 |
| P1（暂缓） | Geometric augmentation | crop、flip、颜色增强暂不锁定 | 第一版不做会改变几何关系的增强；后续若启用必须同步更新 pose、camera、visibility 和 intrinsics |
| P1（100h 正式训练前处理） | Task/text sampling | overfit_v0.0 是单任务数据，不用于判断 100h 多任务频次是否均衡 | 扩展到 100h 时单独审计 task/segment 频次和 caption 长尾；不得反向改变当前 overfit 抽样合同 |
| P1 已锁定 | 训练后人工验收 | overfit 不使用 val/test loss 自动选模 | 按 10.4 节对固定四个 train segments 的每个保存点生成视频、2D hand overlay 和 `F0` 双手/headcam 3D replay |
| P1（暂缓） | Generated-video camera closure metric | TODO-3 双视图人工验收已锁定，自动量化指标尚未锁定 | 不阻塞 smoke test；先保存逐帧 projection overlay 和第三人称 3D replay |

其中 `visibility_gt` 的计算不属于待定项：它必须按第 6 节和独立 TODO-2 文档执行。上述 P0 项完成后，才具备进入 100h 正式 joint training 的最小条件；P1 项可在第一轮 debug/MVP 过程中逐项补齐，但必须记录为实验配置。

参考源码：

- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/data/generator/action/transforms.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/data/generator/action/domain_utils.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/data/generator/sequence_packing/sequence.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/data/generator/sequence_packing/modalities.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/configs/base/experiment/sft/models/nano_model_config.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/model/generator/tokenizers/wan2pt2_vae_4x16x16.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/model/generator/mot/cosmos3_vfm_network.py`
- `/mnt/lzh/cosmos/packages/cosmos3/cosmos_framework/model/generator/algorithm/loss/flow_matching.py`
