# 上游同步记录

- 上游仓库：NVIDIA/cosmos-framework（本目录 `packages/cosmos3/` 对应上游仓库根目录）
- 当前同步 commit：`cf5d68c00d97ccd2480a2320ed652b92dec63102`（Release 2026-09-23）
- 同步日期：2026-09-26
- 此前基线：`5d6dedc`（2026-08-12）
- 同步方式：整体覆盖为上游内容；本地补丁在后续提交中重新移植，并记录在下方"本地补丁"一节。
- v0.4/v0.6 的 joint video-action attention mask 未随同步保留；复现原实验请 checkout `8525625`。

## 本地补丁

以下改动在 cf5d68c 之上重新移植（来源：旧提交 186398c、f695588），保持官方代码结构，改动最小：

- `cosmos_framework/data/generator/action/datasets/action_sft_dataset.py`：`ActionIterableShuffleDataset` 增加 `state_dict` / `load_state_dict`，记录 epoch、分片内样本偏移与 worker RNG，供 `StatefulDataLoader` 精确续训。
- `cosmos_framework/data/generator/joint_dataloader.py`：`RankPartitionedDataLoader` 支持 `stateful=True`（使用 `torchdata` 的 `StatefulDataLoader`）并暴露 `state_dict` / `load_state_dict`；无限 IterableDataset 的 `__len__` 返回 0；`custom_collate_fn` 以列表形式保留 `dataset_index`；`PackingDataLoader` 透传 `prewarm` / `lazy_initialize_child_iterators` 到 `JointDataLoader`，并在 `__iter__` 开头调用 `_initialize_child_iterators_once()`，使 worker 在恢复 dataloader 状态之后才启动。
- `cosmos_framework/trainer/__init__.py`：在 `checkpointer.load()` 之前对提供 `bind_dataloader` 的 callback 调用 `bind_dataloader(dataloader_train)`，使 dataloader 状态 callback 能在 DCP 加载时恢复运行时 loader 的状态。
- `cosmos_framework/callbacks/grad_clip.py`：统计梯度裁剪实际触发情况，在官方 `on_training_step_end` 日志中追加 `grad_clip[/{modality}]/{triggered,trigger_count_window,trigger_count_cumulative,applied_scale}`。每个 optimizer step 只在 GPU 上累计窗口内触发次数（float64 比较，与主机端判定一致）并保留最近一步的 global norm，不调用 `.item()`；仅在 `logging_iter` 对应的日志步同步到 CPU，计算上述四个值，记录数值与原先逐步 `.item()` 的实现相同。
- `cosmos_framework/model/generator/omni_mot_model.py`：cf5d68c 的 `_compute_losses` 直接调用模块级 `compute_flow_matching_loss`，绕过了可覆盖的 `_compute_flow_matching_loss`。action 分支改为调用 `self._compute_flow_matching_loss(...)`（默认实现原样委托，行为不变），使 `EgoVerseOmniMoTModel` 的可见性加权 action loss 继续生效。
