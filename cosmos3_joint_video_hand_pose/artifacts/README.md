# 冻结资产索引

- `cosmos3_hand_codecs/v2_4`：左右独立的冻结 MLP-AE-15；PCA 为历史对照。
- `cosmos3_hand_codecs/v3_wrist_local_pca15_train744`：AR V0.2/V0.3/V0.3.1 的左右独立 wrist-local PCA15，训练集 744 episodes 拟合，保留 manifest、权重和验收记录。
- `cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1`：36 episodes / 181 segments。
- `cosmos3_action_contract/v2`：首帧 state 与旧 future-delta normalizer。
- `cosmos3_action_contract/v4_frame_delta_30hz`：AR V0.1 的 30Hz B3 frame-delta future normalizer，train-only 36-episode 子集统计；资产版本不表示 AR V0.4。
- `cosmos3_action_contract/v1`：历史来源，仍保留在本目录。
- 已归档 `cosmos3_action_contract/v3_frame_delta`：旧版 V0.5/V0.6 B3 future normalizer，迁至 [archive](../archive/artifacts/cosmos3_action_contract/v3_frame_delta/)。
- 已归档 `cosmos3_training_subsets/overfit_100ep_v1`：早期非 AR 联合 WAM 的冻结子集，迁至 [archive](../archive/artifacts/cosmos3_training_subsets/overfit_100ep_v1/)。来源与原因见 [归档索引](../archive/README.md)。

资产版本与实验版本不同。保留原字节及 SHA256，旧 manifest/report 中的绝对路径是生成时的溯源记录，不作为当前仓库根路径。不得通过批量替换改写冻结统计或权重。后续大规模训练需新建版本。
