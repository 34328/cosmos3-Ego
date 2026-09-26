# EgoVerse 纯视频对照

复用联合项目的数据与时间采样规则，训练 moe_gen、time_embedder、vae2llm、llm2vae，冻结动作分支。v1/v2 配方保留在 `configs/`，入口为 `scripts/launch.sh` 与 `scripts/launch_v2.sh`。

已保留的历史 run 是 `outputs/egoverse_it2v/train/pure_it2v_v2_recheck_20260824`，包含 300/600 步检查点，对应回放在 `outputs/egoverse_it2v/inference/` 同名目录。

`scripts/run_replays.sh` 默认 run 名称为 v1，设置 `EGOVERSE_IT2V_VERSION` 可指定已有 run。固定输入默认取已保留的 v0.5 B3 `monitor_inputs`，也可用 `EGOVERSE_MONITOR_ROOT` 指定另一套完整输入。已有回放跳过，不覆盖未完成输出。新输入选择不表示历史纯视频回放已被重新生成。

参见[当前说明](../docs/current-state.md)与[实验归档](../docs/archive/2026-09-26-egowam/README.md)。
