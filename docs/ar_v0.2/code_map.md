# V0.2 代码导航

当前生产代码保留在 `cosmos3_joint_video_hand_pose/src/`，以 `ar_v02_` 区分 V0.1。整理不改变导入路径、模型布局、30 步采样及 checkpoint 合同。

## 按数据流阅读

| 环节 | 模块 | 职责 |
|---|---|---|
| 数据与状态 | `ar_dataset.py`、`ar_chunk_state.py`、`ar_v02_prepare_data.py` | 样本、块首相机系状态、冻结统计 |
| 训练准入 | `ar_v02_dataloader.py`、`dataloader_state.py` | 动态多样本 packing、恢复数据进度 |
| 布局与位置 | `ar_v02_layout.py`、`ar_v02_packing.py` | 显式 U/S/V/A 角色、源帧索引、RoPE、loss 范围 |
| 联合训练 | `ar_v02_model.py`、`ar_v02_attention.py`、`ar_v02_compact.py`、`loss.py` | 双路去噪、可见性、紧凑 query、整体 action loss |
| 模型接入 | `model.py`、`config.py`、`ar_v02_contract.py` | 框架适配、配置注册、续训合同检查 |
| 缓存与采样 | `ar_v02_cache.py`、`ar_v02_inference.py`、`ar_v02_streaming.py` | 15 chunk 历史、离线采样、逐块 API |
| 评测与投影 | `ar_v02_eval.py`、`ar_v02_evaluation.py`、`ar_v02_overlay.py` | CLI、指标与坐标系语义、双栏回放 |

`ar_v02_eval.py` 是命令入口，`ar_v02_evaluation.py` 是指标实现，二者不是重复文件。公共 V0.1 模块仍保留，不能因名称相似而删除。

## 配置、验证和上游补丁

- 正式配置：`configs/ar_v0_2.toml`。
- `configs/ar_v0_2_c.toml`：既有运行记录的兼容别名，不是独立实验。
- `configs/ar_v0_2_c_no_chunk_state.toml`：暂停、不可运行，仅留作历史引用。
- `tests/test_ar_v02_*.py`：按模块组织的回归；`*_gpu.py` 需要 GPU。状态和整体损失另见 `test_ar_chunk_state.py`、`test_whole_action_loss.py`。
- [scripts/README.md](../../scripts/README.md)：Nano 对照、流式 smoke、梯度和恢复验证的入口及使用边界。
- [packages/cosmos3/UPSTREAM.md](../../packages/cosmos3/UPSTREAM.md)：框架补丁清单；不把项目逻辑继续堆入上游目录。
- `outputs/`：结果、日志及诊断快照，不进入 Git；数据和模型权重不随代码整理移动或删除。

## 本次整理验证

2026-09-28 对 47 个本轮新增／修改的项目 Python 文件统一为 120 列格式。逐文件比较格式化前后 Python AST，全部一致。验证脚本另外改为从自身位置解析仓库根目录，Python 子进程沿用当前解释器；GPU 检查与运行保护保留。

此前 4,540 项 Nano 零差异结果属于整理前的源文件指纹；格式化不会使旧指纹变成新指纹。原证据保留，本次新增检查单独记录，不能宣称重跑了完整 Nano 验收。正式训练、50 步和速度验收仍暂停。

整理后验证：V0.2 全部非 GPU 测试，加上块首状态与整体 action loss 测试，共 **330 项通过**（190.28 秒）。47 个文件格式检查、所有待提交 Python 文件语法检查、Shell 语法及默认不执行路径、训练验证脚本 `--help`、Git whitespace 检查均通过。日志位于 `outputs/maintenance/cleanup_20260928/`；本次未重跑 GPU、正式训练或性能测试。
