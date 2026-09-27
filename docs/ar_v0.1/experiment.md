# AR v0.1 实验记录

- 日期：2026-09-27
- 分支：`ar-video-action`
- 方案见 [design.md](design.md)，外部审阅见 [review_codex_2026-09-27.md](review_codex_2026-09-27.md)

**结论**：链路已经跑通，包括训练、逐 chunk 先视频后动作的推理、评测和手部投影回放。
- 视频生成中规中矩。
- 动作在读取真实当前视频时学得好（训练集单 chunk 手腕误差 18 mm）。换成模型生成的视频后，误差回到"不动"的水平。
- 动作质量的改进放到 v0.2，见第 6 节。

## 1. 训练

| 项目 | 值 |
|---|---|
| 初始化 | `/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464`（**不是官方发布权重**，见下方说明） |
| 配置 | `configs/ar_v0_1.toml`（训练时的版本） |
| 节点 / 时长 | Tdebug1，约 1.6 h |
| wandb | [oyeuy4nx](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/oyeuy4nx) |
| 输出（仅保留 `iter_000001200`） | `outputs/joint_video_hand_pose/ar/ar_v0.1_sft48464` |

- **设置**：
  - 数据：36 episode 子集（brushing_shoes / repair_bench），可用 clip 148 个。
  - 硬件与并行：8 卡 H800，CP1/FSDP8。每卡每步 1 个 clip，有效 batch 为 8。
  - 优化：1200 步，lr 2e-5（warmup 100 步，cosine 衰减到 0.1×）。
  - 随机化：每步随机 C ∈ {1,2,3,4}、窗口 ∈ [4,64]；视频和 action 各自按 chunk 独立采样 σ。
  - loss = video + 0.7 × action。
  - 速度与显存：每步约 4.7 s，每卡约 55 GiB。
- **稳定性**：全程无报错，梯度 NaN/Inf 计数为 0。
- **Loss**（把 1200 步均分 7 段，每段取平均，首段 → 末段）：total 0.381 → 0.169，video 0.161 → 0.146，action 0.315 → 0.034。
- **初始化说明**：这个 checkpoint 沿用自仓库早期 v0.x 配置的默认值，不是官方原始权重。逐个参数和官方 Cosmos3-Nano 比对（`outputs/joint_video_hand_pose/ar/ckpt_checks/official_vs_sft48464.json`）：
  - 809 个参数中 416 个完全相同，包括理解通路、词嵌入和 action 投影；
  - 393 个不同，集中在生成通路（`*_moe_gen`、`llm2vae` / `vae2llm`、`time_embedder`），相对差中位数 2.7%，最大 31%；
  - 目录里还带 optimizer、scheduler、trainer 状态。

  所以它是官方 Nano 的生成通路又接着训练过的版本。**之后的训练全部从官方原始 Nano 开始**：即 `/mnt/checkpoints/Cosmos3-Nano` 的 DCP 版本 `/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp`，`configs/ar_v0_1.toml` 已默认加载它。本文数字都来自这次非官方初始化的训练。

## 2. 评测设置

- **权重**：上述训练的 `iter_000001200`。
- **数据**：训练集、验证集各 4 个 T=129 clip，每个 8 个未来 chunk，约 8.6 s 真实时间。
  - **训练集**：36 ep 子集的 index 0/2/4/8。
  - **验证集**：同任务同场景、未参与训练的 4 个 episode，其中 2 个操作者在训练集出现过，2 个是新操作者。清单在 `outputs/joint_video_hand_pose/ar/eval/heldout_manifest/`。原 36 ep 子集全部是 train，没有划分验证集。
- **推理**：C=4，window=30，视频与 action 各 20 步 Euler。第 0 帧（视频首帧 + 首帧 state）给真值。
- **三种条件**（`ar_inference --history`）。每个 chunk 都先生成视频、再生成 action，三种条件的区别在于 action 能看到什么：

| 条件 | 前面 chunk 的历史 | action 读取的当前 chunk 视频 | 与 teacher forcing 的关系 |
|---|---|---|---|
| `oracle` | 真值 | 真值 | 完全 teacher forcing，与训练输入相同，是 action 分支的上限 |
| `gt` | 真值 | 模型刚生成的 | 历史仍是 teacher forcing，只有当前视频换成生成的 |
| `generated` | 模型自己生成的 | 模型刚生成的 | 没有 teacher forcing，即实际部署的情况 |

- **指标**：
  - **单 chunk 误差**：该 chunk 之前用真值，只接上该 chunk 预测的增量，取 chunk 末端的位置误差。手部误差用该 chunk 内 21 点的 MPJPE。这样排除了前面 chunk 累积的漂移。
  - **不动**：参照线，假设该 chunk 内手和头完全不动时的误差，也就是实际移动的距离。
  - **整段漂移**：从首帧开始积分全部预测增量，看整段末端（第 128 帧）的误差。
  - **视频 PSNR**：生成视频与 GT 视频逐帧比较。

## 3. 结果

### 3.1 单 chunk 动作误差（mm）

每格依次为：右手腕末端 / 左手腕末端 / 头部末端 / 右手 MPJPE / 左手 MPJPE。`gt` 和 `generated` 在第 1 个 chunk 的输入相同。

| 条件 | 训练集 chunk 2–8 | 验证集 chunk 2–8 | 训练集 chunk 1 | 验证集 chunk 1 |
|---|---|---|---|---|
| 不动 | 73 / 45 / 47 / — / — | 43 / 24 / 19 / — / — | 84 / 52 / 70 / — | 73 / 29 / 34 / — |
| `oracle` | **18 / 9 / 4 / 14 / 8** | **26 / 18 / 12 / 24 / 15** | 44 / 18 / 6 / 22 | 56 / 20 / 14 / 42 |
| `gt` | 71 / 49 / 55 / 50 / 28 | 56 / 33 / 36 / 44 / 22 | 89 / 64 / 71 / 75 | 69 / 38 / 49 / 56 |
| `generated` | 96 / 54 / 49 / 75 / 44 | 70 / 40 / 41 / 55 / 31 | 同 `gt` | 同 `gt` |

### 3.2 `generated` 整段漂移（第 128 帧，mm）

| | 右手腕 | 左手腕 | 头部 |
|---|---|---|---|
| 训练集 | 150 | 116 | 187 |
| 验证集 | 108 | 109 | 105 |

### 3.3 视频 PSNR（dB，第 1 个 chunk / chunk 2–8）

`oracle` 和 `gt` 用的是同一份生成视频，两者只在 action 读取的视频上不同。

| 条件 | 训练集 | 验证集 |
|---|---|---|
| `oracle` / `gt` | 16.5 / 17.5 | 17.8 / 18.7 |
| `generated` | 16.5 / 13.1 | 17.8 / 14.6 |

### 3.4 手部投影

`src/ar_overlay.py` 的做法：
1. 把 GT 与预测的 57D action 解码为首帧相机坐标系下的 21 点手部。
2. 用 episode 自己的内参 `intrinsics.front_1` 和每帧的 **GT 相机位姿** 投影到画面上，不用预测的相机位姿。
3. 左栏是 GT 视频，右栏是生成视频；两栏都画绿色 GT 手和红色预测手。

投影本身的正确性：GT action 解码后重投影，与直接投影 zarr 原始关键点相比，中位差 ≤ 2.4 px。

预测手与 GT 手的平均像素距离（画面 640×360，两只手、全部 128 帧平均；预测手从首帧积分，包含累积漂移）：

| 条件 | 训练集 | 验证集 |
|---|---|---|
| `oracle` | 22 | 23 |
| `gt` | 61 | 46 |
| `generated` | 71 | 41 |

### 3.5 回放示例

每组第 1 个 clip：训练集为 `69b1c40f…:0`（scrub shoe with brush），验证集为 `69b1d55a…:0`（新 episode）。点击链接在 wandb 查看视频和截图。
- 左栏是 GT 视频，右栏是生成视频。
- 绿色是 GT 手，红色是预测手，都用 GT 相机位姿投影。
- 每张图从上到下是第 16 / 64 / 128 帧，约真实时间 1 / 4.3 / 8.5 s。

| 条件 | 训练集 | 验证集 |
|---|---|---|
| `oracle` | [train oracle](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r) · `showcase/train_oracle.mp4` | [heldout oracle](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r) · `showcase/heldout_oracle.mp4` |
| `gt` | [train gt](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r) · `showcase/train_gt.mp4` | [heldout gt](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r) · `showcase/heldout_gt.mp4` |
| `generated` | [train generated](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r) · `showcase/train_generated.mp4` | [heldout generated](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r) · `showcase/heldout_generated.mp4` |

各条件怎么看：
- `oracle`：红色手紧贴绿色手，对应表 3.1 里的 18 / 26 mm。
- `gt`：右栏生成视频里的手，经常和红色预测手不重合。这说明 action 没有跟上生成的视频，是 3.1 里误差回到"不动"水平的直观表现。
- `generated`：画面一直稳定。但动作逐渐走向和 GT 不同的轨迹，到第 128 帧红色手整体偏下，对应 3.2 的漂移。

在线查看这 6 个视频和图：wandb [ar_v0.1_eval_showcase](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/z64udx0r)。文件在远端 `outputs/joint_video_hand_pose/ar/eval/showcase/`，本地在 `eval_videos/ar_v0.1/`。

全部 4 个 clip × 3 种条件的视频在远端各评测目录下：`<index>_<history>_overlay.mp4`，以及不带手部叠加的 `*_real_time.mp4` / `*_model_time.mp4`。

## 4. 推理耗时

测量条件：单卡 H800，上述 `iter_000001200`，T=129（8 个 chunk），C=4，视频与 action 各 20 步。

推理流程如下：
- 每个 chunk 先做 20 步视频去噪，再做 20 步 action 去噪。
- 每一步都在"截断到当前 chunk 末尾"的片段上重跑一次完整的 teacher-forcing 前向。v0.1 没有 KV cache，所以越往后的 chunk，前缀越长，单步越慢。
- 视频步和 action 步跑的是同一个前向（只取不同的输出），所以两者单步耗时相同。
- 用 `src/ar_benchmark.py` 测量：每个阶段计时 3 步，再换算成 20 步。

各 chunk 耗时 = 20 步视频 + 20 步 action，单位秒。视频与 action 各占一半。

| 分辨率 | 视频 token / latent 帧 | chunk 1 … 8 耗时 | 8 个 chunk 合计 | 单步（chunk 1 → 8） | VAE 编码 / 解码 | 峰值显存 |
|---|---|---|---|---|---|---|
| 320×192 | 60 | 9 / 9 / 9 / 9 / 9 / 9 / 10 / 11 | 73 s | 0.23 → 0.27 s | 0.5 / 0.7 s | 33 GiB |
| 480×288 | 135 | 9 / 9 / 10 / 13 / 14 / 17 / 20 / 22 | 112 s | 0.22 → 0.54 s | 0.4 / 1.3 s | 38 GiB |
| **640×368（训练分辨率）** | 240 | 9 / 11 / 16 / 20 / 24 / 29 / 33 / 38 | **179 s** | 0.22 → 0.95 s | 0.6 / 1.8 s | 45 GiB |
| 960×544 | 510 | 13 / 22 / 35 / 42 / 52 / 65 / 77 / 89 | 395 s | 0.32 → 2.24 s | 1.2 / 3.8 s | 52 GiB |
| 1280×736 | 920 | 单卡 OOM（约 6 万 token 的截断前向超出 80 GiB） | — | — | — | > 79 GiB |

注意：
- 模型只在 640×368 上训练过。其他分辨率只测速度，不代表生成质量。
- 实际评测每个条件还要加上 block mask 构建和解码，640×368 下约 3.5–4 分钟。

- 一个 chunk 对应 16 个视频帧、32 个 action，约 1.07 s 真实时间。在 640×368 下，生成一个 chunk 要 9–38 s，比实时慢约 8–35 倍。
- chunk 耗时随 chunk 序号线性增长，原因是前缀越来越长。有了 KV cache 之后，每个 chunk 的耗时应接近第 1 个 chunk，640×368 下约 9 s。
- 低分辨率时单步约 0.22 s，基本是固定开销。
- VAE 编码和解码的耗时相对去噪可以忽略。

## 5. 结论

1. **看到真实视频时，动作学得好**：`oracle` 下训练集 chunk 2–8 的右手腕误差 18 mm，约为"不动"的 1/4；头部 4 mm，手形 MPJPE 8–14 mm。验证集右手腕 26 mm，低于"不动"的 43 mm。
2. **瓶颈在生成视频到动作这一步**：从 `oracle` 换成 `gt` 后，误差回到"不动"的水平甚至更高。原因有两种可能，目前还分不开：
   - 生成视频不准，动作跟着出错；
   - 生成视频走向了另一个合理的未来，动作和生成视频是一致的，只是和 GT 对不上。
   投影视频的右栏可以直接看生成画面里的手和红色预测手是否重合。要定量区分，需要在生成视频上做手部检测。
3. **第 1 个 chunk 最差**：即使在 `oracle` 下，第 1 个 chunk 的误差也是后面 chunk 的 2–3 倍。可能的原因：
   - 第 1 个 chunk 之前只有首帧 state，没有运动历史；
   - 首帧 state 的时间位置有偏差（见 [design.md](design.md) 第 11 节"已知偏差"）。
4. **动作从首帧开始积分，误差只增不减**：
   - action 除首帧 state 外都是逐帧增量，解码时从首帧开始连乘到末帧。第 k 个 chunk 的起点，就是前面所有 chunk 预测增量累加的结果。
   - `generated` 整段漂移到第 128 帧达到 10–19 cm。
   - 首帧 state 只在第 0 组出现。rollout 更长、首帧超出注意力窗口后，模型就失去绝对位置参考。部署时拿到的新观测也没有位置可以放进去。
5. **视频**：真实历史下 PSNR 17–19 dB。完全自回归时降到 13–15 dB，但画面一直稳定，没有崩坏（见回放视频）。
6. **速度**：见第 4 节。v0.1 的推理只能用于离线评测，要实时必须做 KV cache。

## 6. v0.2 改进方向（动作）

按优先级排列：

1. **先定位问题**：在生成视频上跑手部姿态检测，与预测 action 比较，区分"视频不准"和"动作与视频不一致"。这决定后面该优先改哪一部分。
2. **action 表示**：
   - 把逐帧增量改为相对 chunk 起点的位姿，或在每个 chunk 开头放一次绝对 state。
   - 推理时 state 用预测值，部署时换成真实观测，相当于闭环刷新，同时解决积分漂移和长 rollout 丢失锚点的问题。
   - 与 21 DoF 手部编码、camera pose 是否纳入 action 一起定。
3. **修第 1 个 chunk**：
   - 修正首帧 state 的时间位置；
   - 允许真实历史前缀，部署时本来就有历史观测。
4. **提高动作对生成视频的稳健性**：
   - 训练时给 action 读取的当前视频加噪，即 lingbot 的 `noisy_cond_prob`，原方案中的消融实验 A；
   - 或用 self-forcing，让模型在训练时读取自己生成的视频。
5. **loss 与尺度**：按 Codex review 第七节，先统一校准各维度尺度，再把 8 个分块 loss 合成一个整体 MSE。按 A/B/C 三组对照验证。
6. **数据**：扩大训练数据，正式划分验证集。
7. **推理速度**：实现 KV cache，以当前"截断重算"的实现为参考路径逐步对照。
8. **训练 packing**：v0.1 每步每卡只放 1 个 clip，显卡利用率低，见 `docs/ar_v0.2/README.md`。

## 7. 复现

```bash
cd /mnt/lzh/cosmos-EgoWAM
export PYTHONPATH=$PWD:$PWD/packages/cosmos3 PYTORCH_ALLOC_CONF=expandable_segments:True
# 回放评测：每个 T=129 clip 三种条件约 12 分钟，单卡约 50 GiB。多个 clip 时每个 clip 单独起一个进程
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m cosmos3_joint_video_hand_pose.src.ar_inference \
    --ckpt outputs/joint_video_hand_pose/ar/ar_v0.1_sft48464/checkpoints/iter_000001200/model \
    --toml cosmos3_joint_video_hand_pose/configs/ar_v0_1.toml --output <dir> --indices 0 \
    --history oracle,gt,generated
# 验证集：加 --episodes-manifest / --segments-manifest outputs/joint_video_hand_pose/ar/eval/heldout_manifest/*.csv --split heldout
# 手部投影（CPU）
python -m cosmos3_joint_video_hand_pose.src.ar_overlay --eval-dir <dir> \
    --episodes-manifest outputs/joint_video_hand_pose/ar/eval/heldout_manifest/episodes.csv
# 推理耗时
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 -m cosmos3_joint_video_hand_pose.src.ar_benchmark \
    --ckpt <iter>/model --output <dir>
```

**产物**：`outputs/joint_video_hand_pose/ar/eval/`
- `iter1200_{train,heldout}/`：npz、`*_model_time.mp4`、`*_real_time.mp4`、`*_overlay.mp4`、`run.log`、`overlay_report.json`
- `benchmark_iter1200/benchmark.json`
- `showcase/`：第 3.5 节的 6 个回放视频与截图
