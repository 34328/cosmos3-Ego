# 固定 Pick-and-Place 可视化

此目录保存统一渲染入口、网页模板与固定窗口清单。每次实验沿用同一批训练集和 heldout 样本、同一动作截取区间；只替换 checkpoint 及对应 rollout archive 路径。

`selection.json` 冻结训练集 3 窗、heldout 3 窗及动作区间，附本次 V0.3.0 step3000 的 archive 路径。后续实验复制此清单到独立输出目录，用对应版本的原生 `ar_v03_eval sample` / `ar_v031_eval sample` 生成同窗口、同 seed 的 gt/generated archive；更新 `experiment` 和两个 archive 路径，再调用以下入口。不要改样本、起始帧或短片区间，也不要启动完整指标扫描。

- 只按真实视频选择：有可见的拿起、移动、放下过程，手部位移明显；静止、被遮挡或动作不清楚的窗口不收入。不得按预测效果挑样本。
- 长视频保留完整 17 块的推理历史；短视频从同一次长推理按完整动作区间截取，不是独立短 segment 推理，也不机械截开头或结尾。
- 三栏固定为 GT RGB＋绿色真实骨架、gt-history RGB＋红色预测骨架、generated-history RGB＋红色预测骨架。复用生产包中的原生投影和 overlay。
- 按原始时间轴 30fps 回放。当前 273 帧模型窗口对应 545 帧原视频、约 18.17 秒；预测 RGB 在采样时刻之间保持上一帧，不插帧、不平滑。保留原始时间、帧号和块号标注。
- 默认 seed 42、联合 30 步、σ_small=0.02；样本和区间冻结后各 checkpoint 一致。网页显示实验参数和短片来源。
- 样本编号、起始帧和内部文件路径仅保存在复现清单中，不展示在网页界面。
- 实验设置用直接展开的表格，逐项写明含义：σ_small 是视频/action 写入历史缓存前的低噪声扰动强度；17 块是本次完整长推理长度，短片只是其中动作区间的裁剪，不代表模型长度上限。

从仓库根运行，环境与启动脚本一致：

```bash
PYTHONPATH="$PWD:$PWD/packages/cosmos3" LD_LIBRARY_PATH="" \
  /home/lzh/miniconda3/envs/cosmos3/bin/python -m cosmos3_joint_video_hand_pose.visualization.render \
  --manifest cosmos3_joint_video_hand_pose/visualization/selection.json \
  --output outputs/visualization/<version_checkpoint> --workers 6
```

渲染前检查 CPU 配额、内存、负载和 GPU 占用；CPU 渲染每进程线程 1，不占训练 GPU。已有相同清单的 MP4 可复用；输出目录绑定其他清单时拒绝覆盖，选择新目录。短片边界小自检可通过同一入口的 `--self-check` 执行。

网页与 MP4/JPEG、NPZ 保存在 `outputs/visualization/`，本地同步到 `eval_videos/`；这些实验产物不入 Git。只提交本目录中的代码、模板、说明和固定清单。
