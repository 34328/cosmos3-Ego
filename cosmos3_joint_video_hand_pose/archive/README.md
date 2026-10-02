# 历史代码与冻结资产归档

下表保留原路径与来源，文件通过 `git mv` 原样迁移；资产字节、生成时 manifest/report 中的路径及 Git 历史均保留。

| 原路径（仓库根相对路径） | 本目录内路径 | 来源版本 | 归档原因 |
|---|---|---|---|
| `cosmos3_joint_video_hand_pose/src/inference.py` | `src/inference.py` | 早期非 AR 联合 WAM 的 native inference adapter，起源 `f8bde66`、API 更新 `bc9c9cd` | 已退役的通用推理适配；当前 AR 使用自己的版本化采样入口。 |
| `tests/test_inference_adapter.py` | `tests/test_inference_adapter.py` | 上述 native inference adapter 的回归测试 | 随已退役入口保留测试历史，退出活动测试集。 |
| `cosmos3_joint_video_hand_pose/src/ar_benchmark.py` | `src/ar_benchmark.py` | AR V0.1 benchmark，`8fa6e14` | 仍导入已经删除的 `DEFAULT_TOML`，现有工具入口已损坏。 |
| `cosmos3_joint_video_hand_pose/scripts/check_b3_rigid_pose_roundtrip.py` | `scripts/check_b3_rigid_pose_roundtrip.py` | 旧版 V0.5/V0.6 的 B3 frame-delta 诊断，`599577d` | 针对旧表示的刚体往返检查，保留当时验收工具。 |
| `cosmos3_joint_video_hand_pose/scripts/check_ckpt_load_cf5d68c.py` | `scripts/check_ckpt_load_cf5d68c.py` | 旧版 V0.5/V0.6 的特定 checkpoint 加载诊断 | 绑定历史 checkpoint，并依赖已删除的 `DEFAULT_TOML`。 |
| `cosmos3_joint_video_hand_pose/artifacts/cosmos3_action_contract/v3_frame_delta/` | `artifacts/cosmos3_action_contract/v3_frame_delta/` | 旧版 V0.5/V0.6 B3 的 27D future normalizer，`599577d` | 小数据子集统计的旧动作合同；保留冻结统计及生成脚本。 |
| `cosmos3_joint_video_hand_pose/artifacts/cosmos3_training_subsets/overfit_100ep_v1/` | `artifacts/cosmos3_training_subsets/overfit_100ep_v1/` | 早期非 AR 联合 WAM 的 overfit 子集，`f8bde66` | 旧实验的冻结子集和审计记录，退出当前资产导航。 |

本目录是历史快照。原相对导入与历史路径保持原样，归档脚本不能作为当前训练、推理或验收入口直接执行；需要复现时从 Git 恢复其原路径和相应版本环境。归档测试由本目录的 `conftest.py` 排除默认收集。

AR V0.1/V0.2 的其余代码与产物继续保留；当前训练、checkpoint、运行中评测控制器和跨版本公共基础设施不在本次归档范围。
