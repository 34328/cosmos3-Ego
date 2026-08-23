# Artifacts

当前 overfit_v0.0 运行依赖：

- `cosmos3_action_contract/v2`：active action/normalizer 合同。
- `cosmos3_hand_codecs/v2_4`：冻结 MLP-AE-15 权重。
- `cosmos3_training_subsets/brushing_shoes_repair_bench_36ep_v1`：36 episodes / 181 segments。

保留但不作为当前训练输入：

- `cosmos3_action_contract/v1`：v2 的来源和历史审计依据。
- `cosmos3_training_subsets/overfit_100ep_v1`：早期 100-episode 数据子集，只作历史审计。

这里的 `v1/v2/v2_4` 是 artifact schema 或冻结资产版本，不等于实验版本。
