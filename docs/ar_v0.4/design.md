# AR 联合 V0.4：连续视频 VAE

2026-10-03。状态：实现与必要边界验证完成，未启动正式训练。问题证据见 [公用问题清单](../problems.md)。旧 V0.3 的逐块 VAE 重启是跳变的重要嫌疑，尚未证明是唯一原因。

## 唯一视频语义修正

- 基于最近 V0.3.1：action 57D/PCA15、腕权、状态/坐标系、action loss、前缀采样及分子屏蔽、原分母、C4/K8/H15、数据时序与优化配置保持原样。
- 每个样本调用官方完整视频 VAE encode，然后从同一连续 latent 取各块；保留原 U+4V 网络容器，U 是前一个连续 latent 的引用，不是单帧重新编码。网络容器的 `joint_chunk_cond_v1` 标签不代表视频编码格式相同。
- generated-history 的下一块 U 直接复制上块最后一个预测 latent；保留 action 条件、联合去噪顺序和 video/action 历史 KV refresh。刷新只用于历史写入，不修改输出 latent。
- 去掉重复 U 后整条 latent 调用一次官方 decode，之后才切 RGB 视图供旧 renderer；重叠端点来自同一解码序列，支持 partial 尾块。
- 原视频 stride2、action stride1 和源帧映射保持不变：1 个未来 latent 对应4个采样RGB帧/8个源action帧。没有插帧或动作平滑。gt-history 仍每块提供真实条件，需与自主 generated-history 分开解释。

## 入口与格式

- 训练：`cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_4.sh`，新配置 `cosmos3_joint_video_hand_pose/configs/ar_v0_4.toml`，注册名 `rbs_wam_ar_v0_4_continuous_video_v1`。入口委托官方 Cosmos CLI/Trainer，不另写训练循环。
- 推理：`python -m cosmos3_joint_video_hand_pose.src.ar_v04_eval sample --help`。沿用已有 dataset、action sampler、指标和 renderer；新入口负责连续 codec 与版本验证。
- 视频格式为 `continuous_vae_joint_v1`，模型版本为 `ar_v0.4`。官方成功保存 DCP 后写 `ar_v04_video_format.json`；实际加载来源在官方 load callback 校验。正式初始化只接受已知官方 Nano 路径，恢复/推理要求 V0.4 格式回执。
- 新 NPZ 也必须标记连续格式和整段解码。旧联合 DCP/NPZ 即便 tensor shape 一样也不能直接当连续 latent 使用；不将旧 checkpoint 重命名或强行连续解码。
- 新文件为 `src/ar_v04_{model,config,codec,inference,checkpoint,archive,eval_contract,eval}.py` 与 `src/train_ar_v04.py`，不修改旧联合实现，不影响 `/mnt/lzh/cosmos-ar-it2v` 正在运行的预训练。

## 验证与限制

全部测试使用仓库根及 `packages/cosmos3` 的 PYTHONPATH，CPU 显式关闭 CUDA。Tdebug1 资源检查后执行，GPU0 用于网络测试、GPU1 用于官方 VAE：

| 检查 | 结果 |
| --- | --- |
| V0.4 模型/配置/格式11项＋原V0.3.1模型8项＋实验配置5项 | 24 passed |
| V0.4 codec、partial尾块、原始时间轴、格式拒绝、实际sampler循环（CPU网络/cache替身） | 24 passed |
| 真实 Cosmos 小型网络、真实 noising/loss/history-gradient 三个分支 | 3 passed，29.73秒 |
| 真实训练RGB＋官方Wan2.2 VAE | 连续latent逐位一致、RGB逐字节一致 |

官方 VAE 检查取37个 stride2 采样帧，640×360原图只在底部反射补8像素到640×368，不缩小；bfloat16，官方 `encode_chunk_frames={256:68,480:24,720:8,768:8}`。覆盖两个C4块和一个partial尾块，latent长度10，RGB块长度17/17/5；每段encode/decode各一次，跨块U更新额外VAE调用为0。H800耗时8.93秒，峰值4.26GiB。回执/脚本在 `tmp/ar_v04_real_vae_check.{json,py}`；网络日志在 `outputs/maintenance/ar_v04_validation/network_gpu.log`。

独立review核对了原生归一化、latent/RGB边界、action时间对齐以及官方DCP同步/异步保存回调。未进行新训练、未实跑完整Nano保存/恢复生命周期；这些验证不能证明画质已经改善，正式训练仍需另行确定并授权。

## 已执行的旧产物清理

用户于2026-10-03明确要求删除旧联合V0.3*产物。六节点盘点未发现旧联合训练/评测进程；仅退役Tdebug4旧网页18766/18767，纯视频18768及当前训练保留。

已删除旧V0.3/V0.3.1正式及短测checkpoint、NPZ、视频和缓存，共1,784,049,934,581字节（约1.62TiB文件量），495个文件/目录/符号链接条目；未跟随符号链接。混合短测仅删除v03子树，v02保留。代码、文档、配置、训练日志、小型结果JSON与W&B ID保留；旧重型产物路径已失效。完整清单在 `tmp/ar_v04_cleanup_receipt.json`，状态 `deleted`，清单内残留目标0。
