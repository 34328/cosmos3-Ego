# TODO-4：EgoVerse 视频训练链路 v1

> 状态：P0 合同已锁定，adapter 与训练配置已实现；8 卡 CP=2、FSDP shard=4 下，90K 在第 39 个 optimizer iteration backward OOM，生产 cap 已收紧到 85K，等待长程训练验收。  
> 目的：把 EgoVerse segment 可靠地转换为 Cosmos 原生的 window、packed batch、VAE 输入和 action schema。

## 1. 已验证的硬约束

- 原始视频为 `640x360`（W×H，数组为 `360x640x3`），不做放大或 resize。
- Wan2.2 VAE 要求物理帧数 `T=1` 或 `T=4n+1`；VAE 不会替自定义 adapter 修正任意长度。
- 使用底部反射 padding 8 px，模型画布为 `640x368`。这是 VAE 的空间整除适配，不改变原始内容坐标。送入 Cosmos 的 `image_size` 必须为 `[368,640,368,640]`，让 VAE 保留完整 23 行 latent；若后两项错误填写原始高度 360，Cosmos 的整数除法会把 latent 向下裁成 22 行并只解码出 352 px。
- 每帧空间 latent token 数为 `ceil(368/32)*ceil(640/32)=240`。
- 首帧始终保留；其余模态必须使用与 RGB 完全相同的时间索引。

## 2. Segment 时间适配

对 manifest 的每个 `[start,end)` segment：

1. 先取完整连续帧序列；不跨 segment 拼接，不改变 caption 语义。
2. 若长度不是 `1+4n`，保留第 0 帧，并从 future 中裁掉末尾 `r=(T-1) mod 4` 帧，使 future 数为 4 的倍数。不得用重复帧伪造长度。例如 `T=6 -> 5`、`T=10 -> 9`；删除的是末尾帧，不会打断保留下来的连续时间序列。
3. 若原始 segment 只有 `T=1` 或裁尾后变成 `T=1`（例如原始 `T=2`），它没有 future 监督目标，训练 adapter 应跳过该 segment，并记录 skip reason，不把它作为只有 condition 的正式 joint-training sample。
4. 使用最终 Cosmos JSON prompt 的实际 token 数计算总长度，并要求严格小于 `max_sequence_length=85000`。
5. 完整对齐片段若超 cap，依次尝试保留原始对齐片段的 `80% -> 70% -> 60% -> 50%`。每一档都从原片段独立计算，不做 `0.8×0.7` 式累乘；目标帧数向下对齐为 `1+4n`。
6. 每档均固定保留首帧，并在其余时间范围内等间隔取样；抽帧后的 `conditioning_fps = source_fps × (T_target-1)/(T_aligned-1)`，从而保持 JSON prompt 的真实 duration 不变。
7. 若 50% 档仍无法严格低于 85K，则在 dataset 初始化阶段丢弃该 segment 并记录 `dropped`，不得在 `__getitem__` 或训练中途报错。

同一组 `indices` 必须同步应用于：RGB、57D action、camera pose、wrist/keypoints、MLP-AE codec 输入、visibility 和所有 future metadata。首帧 index 永远为原 segment 的第一个 index。

使用最终 JSON prompt 和 Cosmos tokenizer 的全量 train 审计结果为：32,355 个 segment 中，100%/80%/70%/60%/50% 分别为 31,747/524/36/25/21，50% 仍超限而丢弃 2 个：`6971e4d55bc05ac3cb654a66:0:0:2970`、`6972bf79793ae530061c9f99:0:0:2983`。100-episode 子集的 770 个 segment 全部保留，分档为 748/19/0/1/2。

## 3. Token cap、并行与 native packing

第一轮真实数据配置：`max_sequence_length=85000`、`CP=2`、`FSDP shard degree=4`，8 卡全部参与（`2×4=8`），启用完整 activation checkpointing，并设置 `PYTORCH_ALLOC_CONF=expandable_segments:True`。85K 是 packed-batch 上限，不是固定 segment 长度。

早期单步实测中，约 110K packed batch 可以完成 CP slot 0，但这不足以代表正式训练。105K 的 slot 0 成功、slot 1 backward OOM；90K 随后通过三步基础 smoke，但在 100-episode overfit 的第 39 个 optimizer iteration，CP 对应的 GPU 4/5 各需再申请 962 MiB、仅余 817/899 MiB，最终 backward OOM。这说明三步 smoke 未覆盖最坏 packed batch，90K 不能作为长程训练安全上限。生产 cap 因此收紧为 85K，并要求启动时显式设置 expandable allocator。85K 已通过 8-rank、每 rank 10 个 packed batch 的 dataloader 审计：80 包为 74,904--84,994 tokens，每包 1--13 个 segment，平均每个 optimizer iteration 约消费 18.5 个 segment；GPU 长程训练仍待验收。

Cosmos 原生 `PackingDataLoader` 负责把已经合法且未超 cap 的完整 sample 动态装入一个 packed sequence：长样本可独占 window，短样本可组合；它不会拆分、补齐或挽救超长/非法 `T`。Window sampler、4n+1 对齐和超长均匀采样属于 EgoVerse adapter，不是 packer 的隐式功能。

## 4. VAE 与 batch schema

- VAE tokenizer 冻结；只对底部 padded 的 `640x368` 视频编码。VAE 解码保持 `640x368`，保存正式生成视频和可视化时再裁回顶部原始 `640x360` 内容区。
- 原始 action 为 `[T,57]`，按 Cosmos 原生流程尾部 pad 为 `[T,64]`；`domain_id=3`、`mode="wam"`。
- slot 0 的视频和 action 为 clean condition；future video/action 为 noisy/generated；`sequence_plan` 使用同一个采样后 `T`。
- batch 至少保持视频 `[3,T,368,640]`、raw action `[T,57]`、padded action `[T,64]`、condition mask 和 caption，并保留 packed sample 边界，便于 loss 与回放对齐。

## 5. 验收清单

基础 smoke 已验证：每个样本 `T=1+4n`；所有模态长度和 indices 一致；每个 packed batch严格小于配置 cap；padding 只在底部且 K 不变；五档选择和 dropped 计数正确；单长、短样本组合和 8 卡 CP/FSDP smoke 至少跨过一个完整 CP window。85K 的长程显存稳定性、checkpoint 保存和 resume 仍需验收。

容量验收结论已归档于本节：105K 在 CP slot 1 backward OOM；90K 虽通过三步 smoke，但在第 39 步 backward OOM；当前生产候选为 85K，dataloader 审计已通过，GPU 长程训练待验收。临时 benchmark 文件不作为正式 artifact 保留。
