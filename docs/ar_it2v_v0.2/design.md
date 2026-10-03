# 完整segment与官方动态packing

本轮实现完整segment采样、必要CPU/GPU短测与记录。2026-10-03用户在短测和独立review交付后明确批准按下述新配方正式启动；不恢复旧run。旧V0.1准确标注为“最长97帧的随机短窗口训练”，已停止且保留。正式run资源、路径和在线记录见experiment.md。

## 数据和末块

每个caption segment是一个样本，使用完整[start_idx,end_idx)时间范围及对应text_normalized；不随机裁短、不跨segment，train/test不变。正常段保留全部连续原帧。仅原段超出75008 token预算时，按用户批准的90%→80%→70%→60%→50%顺序尝试确定性均匀采样，选择第一个可装入的最高比例；50%仍装不下才显式剔除并记录原因、数量和原始时长。该策略显式配置为`long_segment_policy='uniform_retention'`，不属于默认的隐含抽帧。

保留帧数为`M=max(2,floor(T*ratio))`，均匀索引始终保留原始首尾，间隔只相差至多1原帧。`effective_fps=(M-1)/(T-1)*original_fps`，使首末时间跨度保持`(T-1)/original_fps`，不倍速回放。caption动作文本不改，描述时长仍为原segment的`T/original_fps`；文本FPS描述及模型`conditioning_fps`使用有效帧率。记录`orig_true_frames`、批准档位`retention_ratio`、实际`M/T`、`effective_fps`和完整`source_frame_indices`。

官方`sft_dataset.py`的`num_video_frames=-1`读取完整视频后仍floor到4N+1，丢掉不对齐尾帧，且没有Zarr接口。因此适配负责Zarr、segment边界/文本与尾部对齐；继续使用官方caption tokenizer、SequencePlan、PackingDataLoader、Trainer、AdamW、LambdaCosine、DCP和W&B。

Wan2.2 VAE要求4N+1：对实际输入的M帧仅追加0–3张末帧，整段连续编码。记录`video_true_num_frames=M`、`video_temporal_padding`和全部真实source indices；解码裁回M帧。首latent为干净条件且loss权重0；完整未来latent权重1，最后latent按有效帧占比r/4计权，分子/有效分母一致。VAE尾latent混合真实和对齐帧，不能逐RGB分离其内部pad成分，此权重不宣称完全消除VAE内部pad影响；原segment最后一帧保留且有监督。

支持部分末C4块；C4/local16、无sink和sigma sampler不变，优化器使用下文用户新选的参数。尾对齐只影响loss有效权重，不改二值条件/noising。packed样本保持隔离。官方sample-level averaging校正各rank样本数，实现全局segment等权；样本内部按有效latent归一。

## 预算与纳入范围

368×640输入（原360高+8底部空间pad），官方spatial16/patch2后每latent为240视觉tokens。精确成本是`240*(1+ceil((M-1)/4))+实际caption tokens+3`；原生Packer预算为严格上限，等于预算也装不下。75008是约75000向128对齐后的预算。

新注册`rbs_wam_ar_it2v_v0_2_ego100h_full_segments`，`max_samples_per_batch=None`、`max_sequence_length=75008`。原生greedy/lookahead不改；显式retention计划完成后，任何仍被纳入却超预算的样本均硬错误，禁止运行时默默缩短、丢弃或OOM后降档重试。`max_num_tokens_after_packing`在当前模型不执行硬截断，实际约束是loader。

长pack的因果mask使用官方`metadata_run_groups/build_block_mask_from_metadata_runs`构造，避免`create_block_mask`的token²稠密bool/int64临时内存。使用原`mask_mod`、同样的sample/frame/text类型和block排序，等价测试逐位通过；只改变构造方式，不改变可见性、packed隔离或AR窗口。

全部train32355段95.5101h、test1180段3.5689h，无短段排除。训练T的p50/p90/p95/max为187/768/1059/2983，test最长3583帧。精确统计原生tokenizer，16进程约4秒完成33535段，代表性串并结果一致。

75008预算下，全清单按真实tokenizer和完整caption预先决定以下档位；时长均指原segment时长，并非抽帧后的帧数除以30。

| 保留比例 | train段数 | train原时长/h | test段数 | test原时长/h |
|---:|---:|---:|---:|---:|
| 100% | 31266 | 79.747815 | 1133 | 2.860287 |
| 90% | 345 | 4.209620 | 13 | 0.157639 |
| 80% | 301 | 4.108491 | 11 | 0.149815 |
| 70% | 234 | 3.611981 | 13 | 0.201241 |
| 60% | 154 | 2.602306 | 7 | 0.118056 |
| 50% | 33 | 0.679519 | 1 | 0.020787 |
| 50%仍超限，剔除 | 22 | 0.550361 | 2 | 0.061074 |

最终train纳入32333段、94.959731h，排除22段、0.550361h（原训练时长0.5762%）；test纳入1178段、3.507824h，排除2段、0.061074h（1.7113%）。纳入样本最大token分别为74935、74928。原始100h数据不等于完整100h均已纳入，更不等于每段都保持30fps。统计与逐段计划保存在`outputs/maintenance/full_segment_retention_75008_20261003/summary.json`及`summary.jsonl`；summary将纳入集token分布与包含剔除段失败尝试的统计分开，不能用后者认定训练仍有超限样本。

启动前验证receipt的policy、75008预算、比例序列以及manifest/tokenizer/逐段记录hash；dataset按已验证计划确定稳定的纳入索引和明确剔除映射。默认`long_segment_policy='error'`仍拒绝全清单超预算；只有显式启用经用户批准的策略才允许上述抽帧和剔除。token覆盖与CPU预检不代表75k GPU承载已通过。

## 恢复、优化与位置

数据checkpoint保存rank/worker/epoch cursor、pending样本、真实源索引、CFG文本、packing预算/lookahead和dataset合同hash。合同包括sample_mode、manifest、retention policy、比例序列、预算及逐段计划hash。预算、清单、window/full模式或retention映射改变时拒绝恢复，避免旧状态的游标被误解释；恢复不得重新按当前随机状态选档或选帧。

恢复训练同时从已有monitor日志还原首次10步loss基线和近期窗口，只读取checkpoint之前的有效分支；历史缺失时明确报错，不能静默换成恢复后的新基线。显存增长窗口仅在同一进程内比较，重启时清空。

用户已选择新正式配置：四节点、每节点8卡，DP 8×4；`max_iter=1500`、每500步保存，官方AdamW `weight_decay=0.01`，全模型峰值lr `1e-4`；官方LambdaCosine `warmup=100`、`cycle_lengths=[1500]`、`f_max=1.0`、`f_min=0.3`，终点lr `3e-5`。AR分块、局部注意力、噪声抽样等原配方保持不变。数据/packing GPU验证短测保留旧`lr=2e-5, weight_decay=0`，不得将短测配置误标为正式配方。用户已在短测和独立review交付后批准正式启动，实际run见experiment.md。

原生W&B早已上传`optim/lr`，旧run API核实2511个点；新增monitor记录本次update前实际各组LR范围，便于找到曲线，不伪造旧实测值。

官方AdamW支持weight decay；官方Nano SFT配置自身也使用`weight_decay=0`，不能仅凭旧值为0认定实现错误。新`.01`是本次用户确认的预训练选择。

RoPE全段和streaming使用相同文本偏移、绝对latent位置及基座fps modulation；正常30fps对应24/30=0.8步距，retention段使用各自effective_fps，没有每块重置。CMD按current_start累加绝对位置，但其Predict2.5关闭fps modulation，不能直接复制频率到Cosmos3。Stage1未发现随机时间offset训练；保留基座并验证坐标/KV。来源：[CMD wrapper](https://github.com/nv-tlabs/cmd/blob/main/cosmos/wrapper.py)、[Stage1配置](https://github.com/nv-tlabs/cmd/blob/main/configs/cosmos/t24_l21_teacher_causal_flow.yaml)。

## 验证入口

长视频推理复用官方有限DualKVCache：当前C4/local16配置4槽（3个历史块和当前写槽），读取仍限最近12 latent，不永久存储整段历史；首帧、绝对位置和部分末块规则保持不变。

- CPU：与launch同一`PYTHONPATH=<repo>:<repo>/packages/cosmos3`，关闭CUDA后`python -m pytest tests`。
- 全段统计：`python -m cosmos3_ar_it2v.segment_statistics --help`；当前策略需显式传`--long-segment-policy uniform_retention --policy-token-budget 75008`；产物不进源码，小型覆盖/hash回执放evidence。
- 75k GPU短测使用独立32条真实segment清单，包含短/中/长段及8条超预算、按70%保留的长段；保留原始边界和文本，不修改正式清单。只验证官方Trainer/packing及显存承载，不据此宣称预训练质量改善。
- GPU真实短测、末块/KV结果见`experiment.md`，不作为新正式run。
