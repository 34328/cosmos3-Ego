# AR V0.3.0 评测计划

日期：2026-10-02。本文仅定义后续评测入口、冻结输入和比较口径，不表示已经启动正式训练或完成任何 V0.3 checkpoint 评测。正式训练须先向用户交付单节点 20 步短测结果，获得本轮正式启动确认。设计以 [design.md](design.md) 为准，结果另写 [experiment.md](experiment.md)。

## 冻结输入

以下路径和 SHA256 已在共享仓库中只读核对。仓库根为 `/mnt/lzh/cosmos-EgoWAM`；数据沿用 `outputs/data_expansion_20260928/episodes.csv`、`segments.csv`，不重划 split 或拟合统计。

| 数据 | 冻结清单（仓库相对路径） | 窗口数 | SHA256 |
| --- | --- | ---: | --- |
| heldout | `outputs/maintenance/video_lr_eval_extended_20261001T2229/heldout8.json` | 8 | `8ac5c99d0a80f963b03858dd9260e12aa3cccead0e0bf4475427bd863ec1f03d` |
| train | `cosmos3_joint_video_hand_pose/configs/overfit16_windows_20261001.json` | 16 | `7e088380b461fce4ea34ec6aeea141d445d1aff6c3acf81f2cbfb401a6bb9ee4` |

`outputs/maintenance/video_lr_eval_extended_20261001T2229/train16.json` 与训练清单内容及 hash 相同。每个清单项含 `sample_id/start/frames/seed`，全部 seed=42、273 帧、17 块；train/heldout 不交叉。训练16窗口只用于正式多任务模型的训练分布诊断，不启动或评测旧过拟合模型。

## 范围与独立输出

仅在正式训练实际产出并核实完整的第 **500/1000/2000/3000** 步 checkpoint 后，分别运行以下三组；每个时间点共32份 NPZ。

| 数据/模式 | 窗口数 | 固定设置 |
| --- | ---: | --- |
| heldout / gt | 8 | C4/K8/H15，联合30步，video/action shift5，CFG1，`sigma_small=0.02` |
| heldout / generated | 8 | 同上；误差保留历史累计漂移 |
| train16 / gt | 16 | 同上；按每块 GT 边界重置 |

1500/2500步保存不自动扩展为评测。sigma 扫描、oracle/pred_history 和其他训练均不是本轮默认派发范围。

输出规划为 `outputs/maintenance/ar_v03_eval_<本次run标识>/step<六位步数>/<split>_<mode>/window<两位序号>/`，CPU指标与报告放同一步的 `metrics/`、`report/`。实际启动时记录最终完整路径、checkpoint SHA/metadata、源码 commit、配置、节点/GPU、冻结清单 hash 和各子任务派发/退出回执。禁止覆盖 V0.2、其他 checkpoint 或其他窗口的输出。

执行前实时检查获准节点的 CPU 配额/负载、内存及 `nvidia-smi` 显存/利用率/进程。可用空闲 GPU 时按窗口拆分，一张卡一个进程；分到其他共享节点时不复制仓库、数据或权重。分片只从原冻结清单按固定序号选择，不改顺序、seed、起点或帧数，派发记录同时保存完整清单 SHA256 与分片 SHA256。CPU后处理按实际空闲资源选并发，限制每worker的 BLAS/OpenMP/OpenCV 线程为1，先验证代表窗口串行/并行一致，不重复启动相同输出任务。

## V0.3 入口

新增入口为 `cosmos3_joint_video_hand_pose.src.ar_v03_eval`，配置为 `cosmos3_joint_video_hand_pose/configs/ar_v0_3.toml`，注册名为 `rbs_wam_ar_v0_3_diffusion_forcing_wrist_weight_v1`。先导入 `ar_v03_config`，官方 TOML loader 实例化 `EgoVerseARV03Model`；推理继续核对旧数据/表示 contract 和全部学习参数，不能用 V0.2 模型替代。

以下是命令模板，并未执行。`AR_V03_CHECKPOINT` 须填本次 checkpoint 的 DCP `model/` 目录，`AR_V03_GPU` 填实时检查后的空闲卡，`AR_V03_OUTPUT` 填不存在的新输出目录；不得使用尚未产生的 checkpoint。`AR_V03_WINDOWS` 选择上表对应冻结清单或已审计的分片，`AR_V03_SPLIT` 为 `heldout/train`，`AR_V03_MODE` 为 `gt/generated`（train只能gt）。

```bash
cd /mnt/lzh/cosmos-EgoWAM
export LD_LIBRARY_PATH='' PYTHONPATH=.:packages/cosmos3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
AR_V03_PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
CUDA_VISIBLE_DEVICES="${AR_V03_GPU:?填写已检查的空闲卡}" "$AR_V03_PYTHON" \
  -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=1 \
  -m cosmos3_joint_video_hand_pose.src.ar_v03_eval sample \
  --ckpt "${AR_V03_CHECKPOINT:?填写本次完整checkpoint}" \
  --toml cosmos3_joint_video_hand_pose/configs/ar_v0_3.toml \
  --episodes-manifest outputs/data_expansion_20260928/episodes.csv \
  --segments-manifest outputs/data_expansion_20260928/segments.csv \
  --eval-windows "${AR_V03_WINDOWS:?填写冻结清单}" \
  --split "${AR_V03_SPLIT:?填写split}" --history "${AR_V03_MODE:?填写模式}" \
  --chunk-size 4 --video-shift 5 --sigma-small 0.02 \
  --output "${AR_V03_OUTPUT:?填写独立输出目录}"
```

采样默认动作shift5、CFG1和联合30步，非零 `sigma_small` 使用 persistent cache。`--no-cache/--verify-cache` 是 sigma=0 的 clean-reference 路径，不能用于本轮 sigma=0.02 的主评测。完成块的预测先保存；refresh只改变历史KV写入，视频/action同步加小噪声，U/S和action padding保持干净。

NPZ保持兼容的 `ar_v02_rollout_v1` schema，metadata增加 `model_version=ar_v0.3.0`、`sigma_small`、`history_video_sigma`、`history_action_sigma`，后三者都为0.02，确保复用指标时正确识别历史噪声。保留原始 GT 关键点、pose、RGB及冻结 normalizer/codec hash；generated 模式的局部坐标转换不消除历史漂移。

CPU动作/PSNR和投影继续复用已验收实现：

```bash
CUDA_VISIBLE_DEVICES='' "$AR_V03_PYTHON" \
  -m cosmos3_joint_video_hand_pose.src.ar_v03_eval evaluate \
  --input "$AR_V03_OUTPUT/0000_gt.npz" --output "$AR_V03_OUTPUT/action_metrics.json"
CUDA_VISIBLE_DEVICES='' "$AR_V03_PYTHON" \
  -m cosmos3_joint_video_hand_pose.src.ar_v03_eval overlay \
  --input "$AR_V03_OUTPUT/0000_gt.npz" --mode real_time \
  --output "$AR_V03_OUTPUT/overlay.mp4"
```

示例假定一窗口gt分片；generated文件名相应为 `0000_generated.npz`。并行分片文件都可能叫0000，聚合必须依据冻结 `sample_id/start` 对齐，而非只按文件名。动作相机/腕部按块末样本加权，手形按action帧数加权；PSNR先合并future采样帧平方误差与元素数再转dB，不能平均各窗口dB。

## 统一光流及旧基线复用

唯一主口径为 [V0.2 experiment §15.2](../ar_v0.2/experiment.md)：每块首条件图→末future图，有效区域缩放320×180，沿用Farneback参数；幅值是全像素平均幅值之比，方向余弦是展平光流整体点积/L2范数乘积。跨窗口/块先累加点积及两侧平方范数再求余弦，不平均逐像素/逐块余弦、不筛运动像素。零GT运动/零范数显式null。保留1–17块曲线及第17块汇总。

可直接调用只读后处理模块 `cosmos3_joint_video_hand_pose.src.ar_v02_endpoint_video_metrics`，无需改V0.2源码；同一命令只能输入一个checkpoint、一个模式、一个split、一个sigma的全部窗口。若分片，每个窗口具体NPZ路径由派发清单提供。

```bash
CUDA_VISIBLE_DEVICES='' "$AR_V03_PYTHON" \
  -m cosmos3_joint_video_hand_pose.src.ar_v02_endpoint_video_metrics \
  --input "$AR_V03_OUTPUT/0000_gt.npz" \
  --workers 1 --output "$AR_V03_OUTPUT/endpoint_video_metrics.json"
```

正式全组聚合将 `--input` 替换为8/16个冻结窗口的实际路径，`--workers` 按资源检查调整。该模块保留光流累加量，支持原始30Hz时间偏移候选{-4,-2,0,2,4}、共同内区RGB MSE；若最优命中边界，仍只称有限网格最优，不擅自扩大范围。

旧基线原地复用，不重新推理、不改写历史NPZ：

| 组别 | checkpoint/run根 | 已有推理存档入口 | 已统一的320×180指标 |
| --- | --- | --- | --- |
| V0.2 lr2e-5 step1000 | `outputs/joint_video_hand_pose/ar_v0_2/formal_fixed_camera_t273_20260930T011945/` | `outputs/maintenance/claude_diag_20260930/formal_step1000/shard0..7/0000_{gt,generated}.npz` | `outputs/maintenance/video_lr_eval_extended_20261001T2229/metrics/baseline_{gt,generated}.json` |
| V0.2全模型lr1e-4 step1000 | `outputs/joint_video_hand_pose/ar_v0_2/video_lr1e4_20261001T140218/` | `outputs/maintenance/video_lr1e4_20261001T140218/eval/video_lr1e4_step1000_gt_oracle/window00..07/0000_gt.npz`；generated位于对应`video_lr1e4_step1000_pred_generated/` | 同一metrics目录的 `lr1000_{gt,generated}.json` |
| V0.2 lr1e-4 step1000 train16/gt | 同上 | `outputs/maintenance/video_lr_eval_extended_20261001T2229/eval/lr1000_train16_gt_first8/`及`lr1000_train16_gt_last8/` | 同一metrics目录的 `lr1000_train16_gt.json` |

完整逐窗口实际路径及旧SHA256见 `outputs/maintenance/video_lr_eval_extended_20261001T2229/cases.json` 与 `report/archive_manifest.json`，不把范围表达式当真实目录。最终同一表列V0.3四个checkpoint和上述V0.2基线，明确数据、模式、历史sigma。旧baseline推理sigma=0，V0.3主评测sigma=0.02，不能隐去这一差异；不用第12/14节旧光流数字替代第15.2口径。

每个checkpoint核实32份存档完整、全部17块、条件/原始GT、两路帧数、清单身份、参数和hash后再记“评测完成”；缺失/失败报告具体范围，不降样本或验收门槛。结果、逐块图、固定前三个heldout窗口视频和输出索引写入 `docs/ar_v0.3/experiment.md`。训练W&B链接必须是本次在线run且API已核实真实step/loss；本计划不生成或推测该链接。
