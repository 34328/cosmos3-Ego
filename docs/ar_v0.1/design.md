# AR 视频–动作联合生成 v0.1 方案

- 日期：2026-09-26
- 分支：`ar-video-action`（基于 main `d90219e`，清理后为 `497723f`）
- 状态：已确认（2026-09-26）；P0–P5 已完成（2026-09-27），实现记录见第 11 节，实验结果见 [experiment.md](experiment.md)。标 **【待定】** 的部分本版不定，标 **【推迟】** 的部分移到后续版本。

## 0. 目标与范围

v0.1 要跑通的是**分块因果自回归**：每个 chunk 先对视频 latent 去噪、再对 action 去噪，之后两者一起写进 KV cache，下一个 chunk 以它们为历史接着生成。

- **目标**：机制跑通、结果可复现。数据先用现有的 36 episode 子集，在上面过拟合，这一版不追求泛化。
- **不在本版范围**：新数据集、新的手部 tokenizer、用真实观测替换 cache 的闭环部署、推理端的 attention sink。见第 9 节。训练时随机 chunk 大小和随机窗口（对齐 lingbot-va）**包含在 v0.1 内**。

## 1. 与现有 v0.6 的区别

| 项目 | v0.6（现状） | v0.1 AR |
|---|---|---|
| 生成方式 | 整段一起去噪 | 按 chunk 自回归，chunk 内部去噪 |
| 视频注意力 | 视频内部双向 | chunk 之间因果，chunk 内部双向 |
| 动作注意力 | A_t 能看 V_≤⌊t/4⌋ 和 A_≤t | 同一 chunk 内动作能看到当前 chunk 的视频，并能看全部历史（见第 4 节） |
| KV cache | 无 | 有，每个 chunk 生成完用干净结果刷新 |
| 框架 | 本地 `packages/cosmos3`（08-07 旧版） | 同步到官方 `cf5d68c`，使用其 `OmniMoTCausalModel` |

官方现状：
- AR 的推理入口 `iter_samples_from_batch_autoregressive` 只支持把 action 当作输入条件（`forward_dynamics` 等模式）。
- 官方的 `wam` 模式能联合去噪视频和 action，但是整段双向，不是 AR。
- **"AR 生成过程中把 action 当预测目标"需要我们自己实现。**

## 2. 【待定】动作向量内容

- **camera pose 是否纳入 action**：【待定】
- **21 DoF 手部如何编码进 action 向量**：【待定】。现在的做法是每只手 wrist 9 维 + MLP-AE-15，需要调研后重新定。

v0.1 对这部分的处理：
- 把动作编码做成可插拔接口：`ActionCodec.encode(raw) -> [T, D]` / `decode`。**【推迟】** v0.1 直接使用 `Action57Builder`，统一接口随 v0.2 动作向量重新设计一起做。
- 模型只依赖 `D ≤ max_action_dim = 64`，以及每个维度的有效 mask。
- **开发和调试阶段先用现有的 57D B3 编码占位**，这样可以直接复用 v0.6 的 normalizer、loss 和回放。编码定下来后只替换 codec，不改模型代码。

## 3. 时间对齐与 chunk 划分（对齐 lingbot-va，frame_stride = 2）

lingbot-va 的做法见 `wan_va/dataset/lerobot_latent_dataset.py:256`：
- 视频按目标 fps 抽帧，`frame_stride` 是相邻两个抽样帧在原视频中的间隔。
- action 保持原始频率、不抽帧，每个 latent 帧挂 `K = 4 × frame_stride` 个 action token（即配置里的 `action_per_frame`：demo 8、RoboTwin 16、Franka 20、LIBERO 4）。
- 首个 latent 帧补 K 个 0。
- lingbot-va 只用机器人数据，没有处理人手速度问题。

lingbot-va 各配置的 chunk 与窗口设置（`wan_va/configs/va_*_cfg.py`）：

| 配置 | frame_stride | K（action_per_frame） | 推理 chunk C（frame_chunk_size） | 每 chunk action 数 | attn_window |
|---|---|---|---|---|---|
| **demo** | 2 | 8 | **4** | 32 | 30 |
| RoboTwin | 4 | 16 | 2 | 32 | 72 |
| LIBERO | 1 | 4 | 4 | 16 | 30 |
| Franka | 5 | 20 | 4 | 80 | 30 |

- 训练时（`train.py`）每步随机 `chunk_size ∈ [1, 4]`、`window_size ∈ [4, 64]`。
- 窗口判断（`model.py` 的 `block_window_mask`）：`|frame_ids[q] − frame_ids[kv]| ≤ window_size`，其中 `frame_ids = latent帧号 // chunk_size × 2`（视频）或 `× 2 + 1`（动作）。一个 chunk 占 2 个 id，所以窗口约覆盖前 2–32 个 chunk。
- 推理时 chunk 固定为配置里的 C，共生成 `num_chunks_to_infer = 10` 个 chunk；`attn_window` 是 KV cache 容量（单位是 chunk 还是条目，待 P3 时核对 `create_empty_cache`）。

v0.1 取 **frame_stride = 2，K = 8，推理 C = 4**，与 lingbot 的 demo 配置一致：

| 项目 | 取值 |
|---|---|
| 源数据 | 视频 30fps，手部和头部位姿 30Hz |
| 视频 | 每 2 个源帧取 1 帧（真实 15fps），共 `T = 4n+1` 帧，经 VAE 得到 `1+n` 个 latent 帧 |
| action | 源帧全部保留，不插值（真实 30Hz）；每个 latent 帧对应 K = 8 个 token |
| 对应关系 | latent 帧 j（j≥1）覆盖抽样后的视频帧 `4j−3 … 4j`，也就是源帧 `8j−7 … 8j`，对应这 8 个源帧的 action |
| 首帧（latent 0） | 视频首帧作为干净条件；8 个 action 槽位放首帧的真实手部状态，也作为干净条件（实现时二选一：state 重复 8 次，或 state 后补零） |
| chunk（推理） | 按 `[1, C, C, …]` 划分，**C = 4** 个 latent 帧 = 16 个视频帧 = 32 个 action，约为真实时间 1.07s（放慢后的标签时间约 2.1s） |
| chunk（训练） | 每步随机 C ∈ {1, 2, 3, 4}（对齐 lingbot）。首个 latent 单独成块；最后一个 chunk 允许不满（lingbot 的 `latent帧号 // chunk_size` 写法天然允许） |
| 窗口（训练） | 每步随机 `window_size ∈ [4, 64]`，单位与 lingbot 相同（frame_id，一个 chunk 占 2） |
| 训练片段 | T 必须为 16m+1（保证推理 C = 4 时 latent 数减 1 能被 4 整除）。按 segment 长度分档：源帧 ≥ 257 用 **T = 129**（8 个 chunk，约 8.5s 真实时间）；129–256 用 T = 65（4 个 chunk）；65–128 用 T = 33（2 个 chunk）；< 65 丢弃。现有 181 个 segment 中 ≥ 65 源帧的有 148 个；中位数 297 源帧，约一半能用 T = 129（精确比例在 P1 统计） |
| token 数 | 每个 latent 帧：视频 240 + action 8，干净 + 带噪两份共约 496。T = 129（33 个 latent）约 16.4K；T = 65 约 8.4K，都远低于 75K 上限 |

**Cosmos 侧需要改的地方**：
- 视频 VAE 和 action 投影是两个独立的编码器。
- RoPE 中 action 的时间位置是按 action 的 fps 单独计算的（`mrope.py`）：每个 action 的步长为 6/fps_action，每个 latent 帧的步长为 24/fps_video。所以 8 个 30Hz 的 action 正好覆盖一个 15fps latent 帧的时间段，位置编码本身不需要改。
- 真正的限制在 packing：代码把"每个 latent 帧的 action 数"直接写成了 VAE 的时间压缩倍数 tcf，涉及以下位置：
  - `packers.py:463` 的 `num_action_tokens_per_supertoken`；
  - `temporal_causal.py` 里的 null token、reshape 和行数校验；
  - `autoregressive.py` 以及 AR 循环里按 `_tcf` 切 action 的地方。
- v0.1 把这个数解耦成独立参数 `action_tokens_per_latent = K`，和 P0 框架同步一起改。
- attention 和 KV cache 是从 pack 的元数据里读 token 数的，不用改。
- Nano 预训练时 K = 4，改成 K = 8 会有分布偏移。不过 action 头本来就要重新学，这个代价可以接受。

## 3.5 人手速度放慢为 0.5×（方案 A：改 fps 标签）

**stride = 2 本身不会改变动作速度**：它只是让视频变稀疏，action 仍然是每个源帧一个、真实 30Hz。放慢要单独用 fps 标签来实现。

Qwen-RobotManip 原文写的是"EgoVerse 降到原帧率的 45%（约慢 2.2×）"，这句话字面上有歧义：如果只是少取帧、每步的时间步长不变，动作反而会变快。能和"变慢"对上的解读，是把时间轴拉长 1/0.45 倍，也就是把 fps 标低。这正是方案 A。

具体做法：
1. **取帧**：去掉现在"整段均匀抽稀，再按 80%/…/50% 降档"的采样方式，改为在 segment 内取一段连续窗口（视频 stride = 2，action stride = 1）。训练时起点随机，评测时起点固定。
2. **fps 标签** = 真实频率 × speed_factor（0.5）：
   - 视频 `conditioning_fps = 7.5`，action `conditioning_fps_action = 15`；prompt 里的 fps 和时长也按放慢后的时间写。
   - fps 只影响 RoPE 的时间位置和 prompt 字段，网络的其他部分不读取它。视频和 action 按相同比例缩放，两者之间的相对对齐不变。
   - **它改变的是什么**：改的是时间语义，模型每一步输出的动作幅度并没有变小。真正的放慢发生在部署端。
   - **局限**：
     - 人手每个 action token 的位移大约是机器人的 2 倍，改标签解决不了这一点，要靠下面的后续方案。
     - 真实画面被标成慢速后，模型看到的相当于慢动作，和 Nano 学到的物理先验有偏差，需要一段适应期。
3. **部署和回放**：action 按 **15Hz** 执行，速度即为人手的一半；下发给机器人控制器（通常 30–50Hz）前再插值一次。回放视频可以同时导出 7.5fps（模型时间）和 15fps（真实速度）两个版本。
4. **按数据源配置 `speed_factor`**：例如 `{"egoverse": 0.5}`。以后接入机器人数据时设为 1.0。
5. **重新统计 normalizer**：action 改成 30Hz 连续采样后，B3 的逐帧增量会比现在抽稀采样时小，需要重新统计并冻结一份新版本（v4）。

**后续（v0.2，方案 B）**：在原始关键点上把 action 插值加密 2 倍（位置用线性插值，旋转用 SLERP；先插值，再过手部 codec），K 从 8 变为 16。这样每一步的动作幅度真正减半，最适合和机器人数据混训。K 的解耦在 v0.1 已经完成，到时候只需要改数据侧。

## 4. 注意力规则

规则参考 lingbot-va 的 teacher-forcing mask（`wan_va/modules/model.py:154`）。

训练时序列里放两份：一份干净的（历史），一份带噪的（预测目标）。chunk k 内部的顺序编号是视频 `2k`、动作 `2k+1`。

| query | 可以看到 | 看不到 |
|---|---|---|
| 带噪视频 V_k | 文本；干净的 (V, A)_<k；V_k 自身 | 当前的 A_k；未来 |
| 带噪动作 A_k | 文本；干净的 (V, A)_<k；**干净的 V_k**；A_k 自身 | 未来 |
| 干净 token | 文本；干净的 (V, A)_≤ 自己所在的块 | 所有带噪 token |

- **已定：v0.1 与 lingbot-va 对齐**，因为 lingbot-va 有开源代码可以对照，Motus2 没有开源。
- **顺序是先视频、后动作**：动作能看到当前 chunk 的视频，方向和逆动力学一致，也和 v0.6 一致。
- 实现时 chunk 内部顺序做成配置项（`intra_chunk_order`，v0.1 只实现 `video_first`），为后续对比 Motus2 的 `action_first` 或 `joint` 留出接口。
- 训练时动作看到的是 GT 视频（teacher forcing）；推理时看到的是刚生成的视频 chunk。
- **窗口**：以上"可以看到的历史"再与 lingbot 的 `block_window_mask` 取交集，即 `|frame_id(q) − frame_id(kv)| ≤ window_size`；文本不受窗口限制。
- 实现方式：官方 chunkwise TF 只支持固定 C，所以 mask 按 lingbot 的 `frame_ids / noise_ids` 写法自己生成（flex attention），chunk 大小和窗口作为每步的参数传入；官方代码只复用 packing、RoPE 和 flex attention 的底层。

## 5. 训练

- **策略**：`video_temporal_causal=True`，teacher forcing（干净 + 带噪两份序列）。每步随机 chunk 大小 C ∈ {1, 2, 3, 4}、窗口 ∈ [4, 64]，同一 batch 内共用一组（lingbot 做法）。官方 `teacher_forcing_frames_per_chunk` 为固定值，不直接使用。
- **action 作为预测目标**：带噪的那份序列里，当前 chunk 的 action 要加噪声并计算 loss，不能像官方那样当干净条件。
- **噪声**：
  - 每个 chunk 为视频和动作各自独立采样 σ。
  - 动作沿用 v0.3 的独立 action noise schedule。
  - 另有可选开关：以 50% 概率给干净的视频历史加 σ∈[0.5, 1] 的噪声（lingbot 的 `noisy_cond_prob`）。v0.1 默认关闭，作为消融实验 A。
- **loss**：沿用 `src/loss.py`：video 权重 1.0、action 权重 0.7，画外手权重 0，visibility mask 保持不变。
- **初始化**：从 Cosmos3-Nano 初始化（实际使用的权重见第 11 节）。
- **并行与预算**：CP1 / FSDP8，75K token 上限，1200 步，每 300 步保存一次（沿用现有配方）。
  - 序列变成干净 + 带噪两份后 token 数翻倍；按第 3 节核算，T = 129 约 16.4K，在预算内。分档 T 导致 batch 内样本长度不同，按现有 packing 拼接（每样本独立 seq_id）。

## 6. 推理

在 AR 推理入口中新增 `image2video_joint_action` 模式。每个 chunk 的步骤如下。

> **【推迟】持久 KV cache**：v0.1 实际实现（`src/ar_inference.py`）不维护 KV cache，每一步去噪都在"截断到当前 chunk"的片段上重跑 teacher-forcing 前向，与训练感受野完全一致但较慢（T=129 单个条件约 4 分钟）。下述第 1–3 步中的 cache 预填、刷新和第 6 步的窗口淘汰移到后续版本，届时以当前实现为参考路径逐步对照。



1. 首帧：文本、V0、首帧 state 一起预填进 KV cache。如果还有真实历史前缀，也一并预填。
2. 对当前 chunk 的视频去噪 N 步（CFG 可选），然后用干净的 V_k 刷新 cache。
3. 对当前 chunk 的动作去噪 M 步，这时动作能看到 cache 里的 V_k；然后用干净的 A_k 刷新 cache。
4. 用 `on_clean_vision_chunk` 边生成边解码，同时输出 action。
5. **chunk 大小固定 C = 4**。**chunk 数量**：离线评测时由 GT 长度决定，公式 `(T_gt−1)/(4C) = (T_gt−1)/16`；做 demo 时手动指定（lingbot 默认 10）。
6. **cache 窗口**：按 lingbot demo 的 `attn_window = 30` 设置，推理窗口语义与训练 `window_size` 一致；单位在 P3 核对 lingbot `create_empty_cache` 后确定。评测 T = 129 时只有 8 个 chunk，窗口基本不起作用，主要为长 rollout 准备。

## 7. 实现分阶段

| 阶段 | 内容 | 关键文件 |
|---|---|---|
| P0 框架同步 | 把 `packages/cosmos3` 升级到官方 `cf5d68c`（官方新增 124 个文件，两边都有但内容不同的有 226 个）。重新移植我们本地的改动，共 1339 行，集中在 `mot/attention.py`、`mot/flex_attention.py`、`mot/cosmos3_vfm_network.py`，以及 `omni_mot_model.py` 里的 2 行。v0.6 专用的 mask 如果会被 AR mask 取代，就只移植必要部分。同时把每个 latent 帧的 action token 数从 tcf 解耦为参数 K（见第 3 节）。验证 Nano 和 v0.6 的 checkpoint 在新框架下都能加载 | `packages/cosmos3/` |
| P1 数据 | 连续窗口取帧（视频 stride = 2，action stride = 1，K = 8，T 按 segment 长度分档取 129 / 65 / 33）；第 0 个 latent 的槽位放 state；按数据源配置 `speed_factor`（fps 标签为视频 7.5、action 15）；抽出 `ActionCodec` 接口；重新统计 normalizer v4 | `src/dataset.py`、`src/temporal.py`、`src/action.py`、`src/codec.py` |
| P2 模型 | 新增 `EgoVerseOmniMoTCausalModel(OmniMoTCausalModel)`：在 TF 路径中把 action 作为去噪目标，实现第 4 节的 mask（含随机 chunk 与窗口），并保留现有 loss 逻辑 | `src/model.py`、官方 `omni_mot_causal_model.py`、`mot/causal_flex_attention.py` |
| P3 推理 | 实现 `image2video_joint_action` 模式和回放脚本 | `src/inference.py` |
| P4 配置 | `configs/ar_v0_1.toml`，对应的 launch 和 replay 脚本，以及 `src/config.py` 中的新实验项 | `configs/`、`scripts/` |
| P5 评测 | 分两种情况评估：单 chunk（只预测一个 chunk）和整段 rollout；复用 `trajectory_metrics.py` 和四格回放 | `src/trajectory_metrics.py`、`src/monitoring.py` |

## 8. 验证

- **单元测试**（CPU）：
  - mask 语义：按第 4 节的表逐项断言；C 取 1–4（含最后一个 chunk 不满）、窗口取边界值时分别检查。
  - packing 的形状和对齐。
  - **一致性**：在 teacher forcing 条件下，AR 逐 chunk 推理（历史用 GT）得到的单步输出，要和训练前向的结果数值一致。这一项用来保证训练和推理的感受野一致。
  - K = 8 的 packing：null 和条件槽位正确，行数校验和 AR 按 chunk 切 action 正确。
  - RoPE：视频标签 7.5、action 标签 15 时，每组 8 个 action 的时间位置落在对应 latent 帧的时间段内；fps 标签乘以 0.5 后，时间位置正好变为原来的 2 倍；prompt 字段与标签一致。
  - 现有 38 项测试继续通过。
- **Smoke**：8 卡跑 50 步，检查 loss 有限、原始 NaN/Inf 计数为 0，checkpoint 能保存并恢复。
- **过拟合**：跑 1200 步。看训练集上 video/action 的 loss 曲线，并对固定的 4 个样本做单 chunk 和整段 rollout 回放。
- **对照**：和 v0.6 整段生成的回放做比较，重点关注长时漂移。

## 9. 后续版本（不在 v0.1 内）

- 动作向量内容：camera pose、21 DoF 编码（第 2 节）。
- 闭环部署：cache 条目加 `is_pred` 标记，用真实观测和真实 state 替换预测结果（参考 lingbot `_compute_kv_cache`）。
- 推理端 attention sink（官方的 `attention_sink_size`）。
- 新数据集和数据合同、验证集划分、完成度 / done 信号。

## 10. 风险

- **P0 合并量大**：我们本地改过的 attention 文件，官方也改过。P0 单独提交，先保证 v0.6 能在新框架上复现，或者至少 checkpoint 能加载、测试能通过，再开始做 AR。
- **从双向切到因果，初期效果会下降**。如果收敛很慢，考虑官方的 `enable_moba`，让双向训练和因果训练交替进行。
- **token 翻倍**：按第 3 节核算在预算内，但随机 C 与窗口让 flex block mask 每步都要重建，编译和构建开销需要在 smoke 时测量。
- **一致性测试与随机 C**：训练的 C 是随机的，推理固定 C = 4。第 8 节的 TF 一致性测试在 C = 4 下做。
- **CP2 下的非有限梯度问题可能在新框架里复现**，所以继续使用 CP1。

## 11. 实现记录（2026-09-27）

分支 `ar-video-action`，提交 `04f43d2` 起。与上文方案的差异和实现细节如下。

- **框架同步（P0）**：`packages/cosmos3` 整体同步到官方 `cf5d68c`（`UPSTREAM.md` 记录上游版本与全部本地补丁）。v0.4/v0.6 的 joint video-action mask 实验连同配置、TOML、启动/回放脚本已删除，原实现见 `8525625`；非 AR 基线为 v0.5。
- **框架补丁（默认关闭，关闭时与官方一致）**：
  - `action_tokens_per_latent`（K）与 VAE tcf 解耦；视频 mRoPE 与 action 的 `base_temporal_compression_factor` 仍用 tcf。K ≠ tcf 时要求 `fps_action = fps_video × K / tcf`，否则报错。
  - `supervise_temporal_causal_actions`：时间因果打包中，非条件帧的 action 组作为带噪、计 loss 的目标。
  - `KVTrainMemoryValue.gen_attention_override`：由调用方计算单视频项的完整视频注意力（GEN 自注意力与视频→文本交叉注意力同一个 softmax）。
- **数据（P1）**：`src/ar_dataset.py`。第 0 组 8 个槽位为首帧 state 重复 8 次（方案第 3 节二选一中的"重复"）。36 ep 子集实际分档：T=129 共 99 条、T=65 共 17 条、T=33 共 32 条、丢弃 33 条。normalizer v4：`artifacts/cosmos3_action_contract/v4_frame_delta_30hz`，按 30Hz 相邻帧增量统计（181 段、79,436 个增量）。
- **注意力（第 4 节）**：`src/ar_attention.py`，按 lingbot-va 的 clean/noisy 规则加 `|block_id 差| ≤ window`，每层一次 FlexAttention；GPU 单测与稠密参考实现逐项对照，数值与梯度一致。
- **训练（P2）**：`src/ar_model.py` 的 `EgoVerseARModel`，基于官方 replayed teacher forcing（Pass 1 干净、Pass 2 带噪，等价于 lingbot 的 [clean, noisy] 单序列）。每步随机 C ∈ {1,2,3,4}、窗口 ∈ [4,64]；视频与 action 各自按 chunk 独立采样 σ；每步 1 个样本（官方 replay 要求）；干净 K/V 保留梯度。EgoVerse 57D loss 抽成 `EgoVerseLossMixin` 复用。
- **推理（P3）**：`src/ar_inference.py`。每个 chunk 先视频后 action，各用 Euler 流匹配求解；每一步在"截断到当前 chunk 末尾"的片段上重跑 teacher-forcing 前向，前面的 chunk 作为干净条件。在 lingbot mask 下这与训练感受野完全一致，因此 v0.1 不单独维护 KV cache（以速度换正确性）。`--consistency-check` 在 Nano、T=65 上验证：同一 chunk 训练布局与推理布局的预测相对误差 ≤ 1.9%（视频）/ 1.2%（action），为 bf16 精度量级。输出 npz、模型时间与真实时间两版对比视频、轨迹指标。
- **配置（P4）**：`configs/ar_v0_1.toml`；`scripts/launch_ar_v0_1.sh`、`scripts/launch_ar_v0_1_smoke.sh`。8 卡冒烟 4 步：loss 有限，显存峰值 41 GiB。
- **训练与评测（P5）**：1200 步已完成（仅保留 `iter_000001200`），推理耗时见实验记录第 4 节（`src/ar_benchmark.py`），结果见 [experiment.md](experiment.md)。
  - `ar_inference --history` 支持三种条件：`oracle`（真值历史 + 真值当前视频供动作读取）、`gt`（真值历史 + 生成的当前视频）、`generated`（完全自回归），并输出逐 chunk 误差。
  - 修复：`gt` 模式原先把前面 chunk 的预测覆盖成真值（`dae9842`），现在每个 chunk 的预测单独保存。
  - 推理按官方推理模式加载 bf16 权重（不保留 fp32 主副本），T=129 单卡约 50 GiB。
  - 验证集：原 36 ep 子集全部为 train；评测另取同任务同场景、未参与训练的 4 个 episode（`outputs/joint_video_hand_pose/ar/eval/heldout_manifest/`）。这些 episode 没有预先生成 `palm_in_fov` 字段，数据集加载时按同一投影现算。
  - 手部投影：`src/ar_overlay.py`，用 episode 内参和 GT 相机位姿把 GT / 预测手部投到 GT 视频与生成视频上。
- **初始化权重**：v0.1 这次训练用的是 `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464`（沿用自仓库早期 v0.x 配置的默认 `BASE_CHECKPOINT_PATH`），不是官方原始权重：与官方 Cosmos3-Nano 相比，理解通路完全相同，生成通路（`*_moe_gen`、`llm2vae` / `vae2llm`、`time_embedder`）不同，相对差中位数 2.7%，比对结果见实验记录第 1 节。**之后的训练全部从官方原始 Nano 开始**：官方 HF 权重 `/mnt/checkpoints/Cosmos3-Nano` 用官方 `convert_model_to_dcp` 离线转成 DCP，放在 `/mnt/lzh/checkpoints/Cosmos3-Nano-dcp`（转换脚本 `/mnt/lzh/checkpoints/convert_nano_offline.py`：节点无外网，把 Qwen3-VL tokenizer 配置和 Wan VAE 指向 `/mnt/checkpoints` 下的本地文件）。用 AR 模型加载：缺失 0、形状不匹配 0，只多出 5 个声音模块参数（本配置不用），前向正常。
- **已知偏差（留到 v0.2 修）**：
  - 首帧 state 的时间位置：第 0 组 8 个槽位放首帧 state 后，action 行数等于 `T×K`，官方打包据此按"AR 接续"处理，使视频整体时间后移一帧、同一个首帧 state 被编码为首帧之前的 8 个时刻（Codex review 一.3）。未来视频与 action 的相对对齐不受影响，训练与推理一致；改动会改变位置编码，需要重训。
  - 一致性检查只输出误差数字，没有判定阈值；smoke 只跑 4 步，未验证 checkpoint 保存—恢复。
- **推迟到后续版本**：推理端持久 KV cache（第 6 节）、`ActionCodec` 统一接口（第 2 节）、闭环真实观测替换、attention sink（第 9 节）。
