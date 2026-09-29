# V0.2 验证脚本

历史验证入口默认使用 `configs/ar_v0_2.toml`（旧动作表示）；支持 `--toml` 的入口可显式选择新方案 `configs/ar_v0_2_fixed_camera.toml`。新方案的 AE 与统计尚未通过准入，旧配方测试成功不能当作新方案通过。重复 `_c`／`_multitask` TOML 已合并，旧运行快照不修改。

正式数据、训练和评测实现位于 `cosmos3_joint_video_hand_pose/src/`。本目录仅保存可复现的验证入口，不参与模型运行时导入；暂时诊断结果保存在被 Git 忽略的 `outputs/`。

| 文件 | 用途 | 边界 |
|---|---|---|
| `verify_ar_v02_native_training.py` | 官方 CLI 路径检查及可选短步保存／恢复检查 | 默认仅检查；缺新 AE／统计时明确阻塞，脚本测试通过不等于官方 Trainer 生命周期已通过 |
| `check_ar_v02_nano_cache.py` | Nano 缓存与完整前缀重算逐步数值对照 | `--smoke`、`--focus-boundaries` 仅代表指定用例，不能代替完整验收 |
| `ar_v02_nano_validation.py` | 上述对照的公共夹具与参考路径 | 仅测试进程使用；复用 mask，不复用参考路径激活 |
| `diagnose_ar_v02_cache_precision.py` | 定位精度、注意力顺序及分块差异 | 诊断模式不是生产推理设置 |
| `smoke_ar_v02_streaming.py` | 检查逐块 API 与残块连接 | 不代表生成质量或速度验收 |
| `validate_ar_v02_training.py` | 固定输入两次短测；可选 50+1 保存恢复 | 数值夹具不替代官方 Trainer.train 的正式生命周期验收；保留空闲 GPU 检查 |
| `check_ar_v02_resume.sh` | 两次独立进程的 2+1 保存恢复 | 默认仅说明用法，必须显式传入 `--run-authorized`；可用获准节点，仍检查空闲 GPU |

从仓库根目录、已核实的 Cosmos Python 环境调用。Python 脚本使用 `PYTHONPATH=.:packages/cosmos3`、`LD_LIBRARY_PATH=''`；训练验证子进程沿用启动脚本的 Python。Shell 恢复脚本允许通过 `AR_V02_PYTHON` 指定解释器，默认仍为已验证的远端环境。仓库根目录由脚本位置推导。

运行任何 GPU 检查前先检查节点资源；不得终止其他任务。每次使用独立输出目录。当前只保留短测证据，不启动正式训练、50 步或延迟验收。

结果和已知缺口见 [复审记录](../docs/ar_v0.2/experiment.md)，生产模块关系见 [代码导航](../docs/ar_v0.2/design.md#7-代码导航)。

正式训练使用版本化 `cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_2.sh`，source 官方 `_sft_launcher_common.sh`，由项目薄入口转交 Cosmos3 官方 CLI；这些验证脚本不作为新的训练引擎。正式启动前需核实 online 和认证，启动后实际检查在线 step/loss；不得复用 outputs 下的旧临时启动脚本。
