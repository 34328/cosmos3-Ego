# 上游同步记录

- 上游仓库：NVIDIA/cosmos-framework（本目录 `packages/cosmos3/` 对应上游仓库根目录）
- 当前同步 commit：`cf5d68c00d97ccd2480a2320ed652b92dec63102`（Release 2026-09-23）
- 同步日期：2026-09-26
- 此前基线：`5d6dedc`（2026-08-12）
- 同步方式：整体覆盖为上游内容；本地补丁在后续提交中重新移植，并记录在下方"本地补丁"一节。
- v0.4/v0.6 的 joint video-action attention mask 未随同步保留；复现原实验请 checkout `8525625`。

## 本地补丁

### AR V0.2 薄启动入口（2026-09-29）

- `examples/_sft_launcher_common.sh` 增加可选 `WORKDIR`、`TRAINING_MODULE`、`TRAINING_PYTHONPATH`，使仓外项目配置能复用官方检查、torchrun 参数、日志和退出码处理；未指定时保留官方 examples 默认行为。
- 可选 `TEXT_TOKENIZER_PATH` 相对路径以工作目录解析并导出。项目 wrapper 保留独立 tokenizer 默认目录，不根据目录名猜测与 DCP 是否匹配。
- 不增加训练循环，不改 Trainer／优化器／调度器。CPU 配置与启动测试不代替真实 GPU 保存恢复验收；实际证据见 `docs/ar_v0.2/experiment.md`。

### AR v0.2 逐块图像／state 条件与多样本（2026-09-28，验收中）

- 新布局为 `joint_chunk_cond_v1`：每块独立编码 `U/S/V/A`，替代下节旧单首帧布局。项目层改动位于 `cosmos3_joint_video_hand_pose/src/ar_v02_*`，旧布局的测试结果不能作为新布局验收依据。
- `model_config.py`、`omni_mot_model.py`、`cosmos3_vfm_network.py`：新增默认关闭的 `enable_vision_condition_embedding`，仅对块首图像注入零初始化可学习类型向量；与 `action_state_embed` 一起进入优化器。
- `sequence.py`：新增 `vision_condition_type_mask` 并随 pack 迁移设备；它区分 U/V 类型，不替代 attention mask 或 loss mask。
- `causal_attention.py`：有显式 GEN override 时，不再把两个视频项误判成 control/target transfer；项目层另修 teacher-forcing memory 的逐样本文本 offsets，保证多 clip 文本隔离。
- `cosmos3_vfm_network.py`／`causal_attention.py`：纯文本预填显式使用空 GEN 索引，支持 `flat_gen_tokens=0`，避免未给视频时的 `.to(None)` 和空 GEN 重塑错误。
- 项目层沿用官方动态 packer 和 replay，预算覆盖条件 token 与两次前向；loss 按模态有效样本全局平均，不能对各卡的均值直接平均。官方初始化与续训采用不同的 checkpoint 校验，续训绑定布局、有效窗口及归一化统计。
- 项目层新增 `ar_v02_compact.py`：第二遍仅计算未来 V/A query，复用带梯度的 clean 文本与 GEN K/V；完整 query 路径保留作参考。packer 准入暂保留双完整 pass 的保守预算，未依据节约量扩大 batch。


### AR v0.2 单 state 与显式 joint 布局（2026-09-28，开发中）

- `configs/base/defaults/model_config.py`、`model/generator/omni_mot_model.py`、`mot/cosmos3_vfm_network.py`：新增默认关闭的 `enable_action_state_embedding`。开启时 state 复用 action 投影，并增加零初始化的可学习类型向量；只在 `action_state_mask` 对应行注入，未来 action 的输入／输出维度不变。
- `data/generator/sequence_packing/sequence.py`：`PackedSequence.action_state_mask` 显式记录 state 行，并参与设备迁移、clean replay 深拷贝；state 的 condition／noisy／loss 范围由项目 V0.2 packer 构建。
- `mot/causal_attention.py`：允许 GEN override 声明 `flat_gen_tokens`，不再强迫可变 state 布局重塑为 `T×(K+HW)`；保留真实视频几何，支持无视频的 state-only prefill。持续缓存读取已缓存文本时跳过文本 query，后续前向仅投影当前 GEN token。
- 上述字段默认关闭／为空，V0.1 与非 AR 打包保持原有路径。项目层 `ar_v02_*` 提供显式角色／源帧索引、训练 joint mask、30 步采样、单 state 初始预填、15 chunk 淘汰和显式解包。旧版重复 8 次属于打包实现选择，不是 flow matching 要求。
- 当前 torch 2.10 环境下，编译 `create_block_mask` 并交替使用 clean/noisy 布局会出现前向一致、文本 K 梯度不一致；V0.2 使用 eager mask 构建，FlexAttention 前后向仍编译。覆盖交替布局的 GPU 梯度测试必须保留；未验证前不能重新开启 mask 构建编译。

以下为之前移植的补丁：

以下改动在 cf5d68c 之上重新移植（来源：旧提交 186398c、f695588），保持官方代码结构，改动最小：

- `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py`：`ActionIterableShuffleDataset` 增加 `state_dict` / `load_state_dict`，记录 epoch、分片内样本偏移与 worker RNG，供 `StatefulDataLoader` 精确续训。
- `cosmos_framework/data/generator/joint_dataloader.py`：`RankPartitionedDataLoader` 支持 `stateful=True`（使用 `torchdata` 的 `StatefulDataLoader`）并暴露 `state_dict` / `load_state_dict`；无限 IterableDataset 的 `__len__` 返回 0；`custom_collate_fn` 以列表形式保留 `dataset_index`；`PackingDataLoader` 透传 `prewarm` / `lazy_initialize_child_iterators` 到 `JointDataLoader`，并在 `__iter__` 开头调用 `_initialize_child_iterators_once()`，使 worker 在恢复 dataloader 状态之后才启动。
- `cosmos_framework/trainer/__init__.py`：在 `checkpointer.load()` 之前对提供 `bind_dataloader` 的 callback 调用 `bind_dataloader(dataloader_train)`，使 dataloader 状态 callback 能在 DCP 加载时恢复运行时 loader 的状态。
- `cosmos_framework/callbacks/grad_clip.py`：统计梯度裁剪实际触发情况，在官方 `on_training_step_end` 日志中追加 `grad_clip[/{modality}]/{triggered,trigger_count_window,trigger_count_cumulative,applied_scale}`。每个 optimizer step 只在 GPU 上累计窗口内触发次数（float64 比较，与主机端判定一致）并保留最近一步的 global norm，不调用 `.item()`；仅在 `logging_iter` 对应的日志步同步到 CPU，计算上述四个值，记录数值与原先逐步 `.item()` 的实现相同。
- `cosmos_framework/model/generator/omni_mot_model.py`：cf5d68c 的 `_compute_losses` 直接调用模块级 `compute_flow_matching_loss`，绕过了可覆盖的 `_compute_flow_matching_loss`。action 分支改为调用 `self._compute_flow_matching_loss(...)`（默认实现原样委托，行为不变），使 `EgoVerseOmniMoTModel` 的可见性加权 action loss 继续生效。
- `action_tokens_per_latent`（K）与 VAE 时间压缩倍数 tcf 解耦：temporal-causal packing 中每个 latent 帧对应的 action token 数改为独立参数 K，默认 `None` 即 K=tcf，旧行为逐元素不变；mRoPE 时钟单位仍为 tcf（视频 `temporal_compression_factor=tcf`、action `base_temporal_compression_factor=tcf`），`mrope.py` 未改。下游 attention / NATTEN / KV cache 读取 packing 写入的 `num_action_tokens_per_supertoken`，无需改动。涉及文件：
  - `cosmos_framework/configs/base/defaults/model_config.py`：`OmniMoTModelConfig.action_tokens_per_latent: int | None = None`。
  - `cosmos_framework/configs/toml_config/sft_config.py`、`toml_config_helper.py`：TOML `[model].action_tokens_per_latent`（VFM 映射到 `model.config.*`，VLM 跳过）。
  - `cosmos_framework/data/generator/sequence_packing/temporal_causal.py`：`pack_supertokens_temporal_causal(action_tokens_per_latent=...)`；null token、行数校验、token_shapes / condition_mask、逐帧 action span 均按 K；K≠tcf 且给定两个 FPS 时要求 `fps_action ≈ fps_video * K / tcf`，否则 raise。
  - `cosmos_framework/data/generator/sequence_packing/packers.py`：`pack_input_sequence(action_tokens_per_latent=...)` 透传，`num_action_tokens_per_supertoken` 写入 K。
  - `cosmos_framework/data/generator/sequence_packing/autoregressive.py`：两个 AR pack 函数新增参数并透传；`action_domain_id` 期望数量按 K 计算。
  - `cosmos_framework/data/generator/sequence_packing/sequence.py`、`types.py`：仅更新 `num_action_tokens_per_supertoken` 字段注释。
  - `cosmos_framework/model/generator/omni_mot_model.py`：`_pack_input_sequence` 透传 `config.action_tokens_per_latent`；无 `conditioning_fps_action` 且 K≠tcf 时 `fps_action = conditioning_fps * K / tcf`。
  - `cosmos_framework/model/generator/omni_mot_causal_model.py`：chunkwise TF 截断、AR 推理中 action 切片 / streamed action 形状校验 / domain id 切片改用 K；各 AR pack 调用与 `_seed_frame_into_kv_cache` 透传 K；batched streaming Transfer 路径的占位 `fps_action_list` 在 K=None 时保持 24.0，否则取 `conditioning_fps_action` 或 `fps_video * K / tcf`。
  - `cosmos_framework/model/generator/omni_mot_causal_model_test.py`：`TestARGenerationLoopLogic` 的 MagicMock 配置显式设置 `action_tokens_per_latent = None`。

### 联合视频–动作 teacher forcing（AR v0.1）

默认关闭，关闭时与上游行为一致。

- `configs/base/defaults/model_config.py`、`configs/toml_config/sft_config.py`、`configs/toml_config/toml_config_helper.py`：新增 `supervise_temporal_causal_actions`（默认 False）。
- `data/generator/sequence_packing/temporal_causal.py`、`packers.py`、`model/generator/omni_mot_model.py`：`supervise_action_tokens=True` 时，非条件帧的 action 组成为带噪、计 loss 的目标（写入 condition mask、noisy_frame_indexes、mse_loss_indexes 与逐帧 timestep），条件帧的 action 组保持干净。
- `model/generator/utils/kv_cache.py`、`model/generator/mot/causal_attention.py`：`KVTrainMemoryValue.gen_attention_override`（默认 None）；非 None 时由它计算单视频项的完整视频注意力（GEN 自注意力与视频→文本交叉注意力在同一个 softmax 中，不经过 LSE merge），文本自注意力不变。
### V0.2 action bias 梯度汇总（2026-09-28）

`model/generator/mot/domain_aware_linear.py`：grouped bf16 投影的重复 domain bias
先用 FP32 gather，再转回输出 dtype；forward 值和 checkpoint key 不变，反向先累加
再做 bf16 舍入。修复删除零 loss 条件行导致 bias 梯度变化的问题。
回归：`tests/test_ar_v02_domain_bias_gpu.py`、`tests/test_ar_v02_compact_gpu.py`。
