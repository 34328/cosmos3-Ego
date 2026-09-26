# 冻结资产索引

- `cosmos3_hand_codecs/v2_4`：左右独立的冻结 MLP-AE-15；PCA 为历史对照。
- `cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1`：36 episodes / 181 segments。
- `cosmos3_action_contract/v2`：首帧 state 与旧 future-delta normalizer。
- `cosmos3_action_contract/v3_frame_delta`：v0.5/v0.6 B3 future normalizer，仅由小数据子集统计；首帧仍用 v2 state normalizer。
- `cosmos3_action_contract/v1` 与 `cosmos3_training_subsets/overfit_100ep_v1`：历史来源。

资产版本与实验版本不同。保留原字节及 SHA256，旧 manifest/report 中的绝对路径是生成时的溯源记录，不作为当前仓库根路径。不得通过批量替换改写冻结统计或权重。后续大规模训练需新建版本。
