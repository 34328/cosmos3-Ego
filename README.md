# Cosmos EgoWAM / RBS WAM 研究基础

本仓库保留基于 Cosmos 3 的 EgoVerse 视频与双手动作联合训练实现，作为后续 RBS WAM 工作的基础。已有证据来自小数据实验，尚不代表大规模训练或泛化验收。

## 阅读入口

- [当前实现与运行说明](docs/current-state.md)：数据、模型、入口、路径与已知限制。
- [历史实验归档](docs/archive/2026-09-26-egowam/README.md)：实验结论、证据位置及原始文档快照。
- [联合训练文档](cosmos3_joint_video_hand_pose/docs/README.md)。
- [纯视频对照](cosmos3_egoverse_it2v/README.md)。

## 目录

- `cosmos3_joint_video_hand_pose/`：联合训练、动作表示、回放及冻结资产。
- `cosmos3_egoverse_it2v/`：纯视频对照。
- `packages/cosmos3/`：包含本项目修改的 Cosmos 框架源码。
- `training_manifests/`：本机保存的数据清单。
- `outputs/`：训练检查点、日志和回放，不纳入 Git。
- `tests/`：本机联合训练测试，当前被 Git 忽略；纯视频及框架部分测试已纳入 Git。

远端操作使用 MCP Apply SSH。运行脚本从自身位置计算仓库根目录，Python 配置从 `__file__` 计算路径，不再依赖旧 `/mnt/lzh/cosmos` 目录。数据、预训练模型与 Python 环境仍使用现有外部路径。
