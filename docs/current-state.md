# 当前实现与运行说明

核对日期：2026-09-26。仓库位置：`/mnt/lzh/cosmos-EgoWAM`。历史实验结论见[归档](archive/2026-09-26-egowam/README.md)，后续新实验另设名称和输出目录。

## 实现边界

输入为文本、首帧 RGB 和首帧真实双手姿态；同一个 Cosmos Generator 联合预测未来视频、相机和双手姿态。使用 Cosmos3-Nano SFT 初始化；Reasoner、视频 VAE、左右手 MLP-AE-15 冻结，生成路径参与训练。

动作宽度为 57D，尾部补齐到 64D：camera 9、右 wrist 9、右 hand latent 15、左 wrist 9、左 hand latent 15。当前仍是 T 个 action，首帧 a0 为 clean condition。无首帧动作的 T-1 合同只是设计稿。

v0.5 B3 使用逐帧 SE(3) camera/wrist 增量，手型 latent 不做时间差分；更早版本使用首帧相对增量。必须按原版本解码，不能混用 normalizer。v0.4/v0.6 的 joint video-action attention mask 已删除（原实现见 `8525625`），v0.6 checkpoint 仍可作为权重加载。

## 数据与实验参数

Mecka 母集为 4,295 episodes / 33,535 segments；train 为 4,142 / 32,355，test 为 153 / 1,180，无独立 val。现有联合实验使用 brushing_shoes / repair_bench 的 36 episodes / 181 train segments。

视频由 640×360 底部 reflect-pad 到 640×368，T=4n+1；RGB、动作和 visibility 同步采样。当前文本拼接 episode 描述与 segment 指令。联合实验使用 CP1/FSDP8、75K token cap、BF16、full activation checkpoint。video/action loss 权重 1.0/0.7，八个 action 子块等权，画外手权重 0。v0.3 起启用 active normalization 和独立 action noise schedule。

现有配方运行 1200 步、每 300 步保存；base LR 2e-5，shared/video 4 倍、action projection 5 倍，warmup 100。以上是已有小数据配方，不是后续大规模训练的默认批准参数。

## 环境与入口

```bash
cd /mnt/lzh/cosmos-EgoWAM
export PYTHONPATH="$PWD:$PWD/packages/cosmos3"
export LD_LIBRARY_PATH=''
PYTHON=/home/lzh/miniconda3/envs/cosmos3/bin/python
# 仅校验冻结资产，不启动训练
"$PYTHON" cosmos3_joint_video_hand_pose/artifacts/cosmos3_action_contract/v2/validate_manifest.py
```

预训练权重：`/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464`；VAE：`/mnt/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`。数据路径由 episode CSV 中的 `abs_zarr_path` 指定。

联合训练配方位于 `cosmos3_joint_video_hand_pose/configs/*.toml`，启动入口为该项目 `scripts/launch_overfit_*.sh`，配套回放为 `scripts/run_*replay*.sh`。历史 YAML 仅为描述快照，不作为启动依据。已有输出会触发部分脚本的防覆盖检查；新实验先建立独立配方，不复用旧 run 名称。启动前检查 GPU，现有 joint 配方需要 8 张空闲 GPU。

## 已知限制

- joint CP2 曾出现非有限梯度，沿用 CP1；同时查看原始 NaN/Inf 计数，不能只看清洗后的梯度范数。
- B3 future normalizer 仅由 36-episode 子集统计；扩展训练需重新定义并冻结数据合同。
- normalizer/codec 的完整 checkpoint 自动绑定仍待完善。
- 回放是固定训练样本，不能用于宣称泛化。文档中的视觉结论本次未重新人工验收。
- 根目录测试未跟踪；复现所需数据、环境与本地测试不会随普通 Git 克隆自动取得。

## 路径清理记录

训练/回放脚本和 Python 配置已使用动态仓库根目录。旧回放 JSON 中的项目根路径已迁移；原文、逐文件 SHA256 记录保存在 `outputs/maintenance/2026-09-26-path-relocation/`。检查点、日志和冻结 artifact 未重写，内部历史路径仍是溯源信息。再次移动仓库时，已有回放 JSON 需要再次迁移，脚本和配置不需要改根路径。

## ar-video-action 分支清理（2026-09-26）

本分支用于 AR 视频与动作联合生成，清理前的完整状态见 main `d90219e`。

- 移除官方附带的 `evaluation/`、`cookbooks/`，参考源码改用共享路径 `/mnt/lzh/refs/cosmos-framework`。
- 移除纯视频对照 `cosmos3_egoverse_it2v/`，以及 v0.0–v0.5 的 TOML、启动与回放脚本；`src/config.py` 中的继承链保留。
- `tests/` 纳入 Git。
- `outputs/` 仅保留 v0.2 与 v0.6 的 `iter_000001200` 模型权重（不含 optimizer），其余 checkpoint 与本地 W&B、dataloader trace 已删除，删除清单见 `outputs/maintenance/2026-09-26-checkpoint-prune/`。训练曲线以远端 W&B 为准。

## AR v0.1 方案（2026-09-26）

- 方案文档：`docs/ar_v0.1_design.md`（已确认，动作向量内容与 camera pose 仍为【待定】）。
- 要点：对齐 lingbot-va demo 配置，frame_stride = 2、每 latent 帧 K = 8 个 action、推理 chunk C = 4；训练随机 C ∈ [1, 4]、窗口 ∈ [4, 64]；T 按 segment 长度分档 129 / 65 / 33；人手速度通过 fps 标签 × 0.5 放慢（视频 7.5、action 15）。
- 下一步：P0 框架同步官方 cosmos-framework `cf5d68c`，并把 action_tokens_per_latent 从 tcf 解耦。

## P0 框架同步（2026-09-27）

- `packages/cosmos3` 整体同步到官方 `cf5d68c`，上游版本与本地补丁清单见 `packages/cosmos3/UPSTREAM.md`。
- 本地补丁仅保留：dataloader 断点续训、grad clip 触发统计、`omni_mot_model` action loss 走子类覆盖、`action_tokens_per_latent`（K）与 tcf 解耦。
- v0.4/v0.6 的 joint video-action mask 实验连同 v0.6 TOML 与启动/回放脚本删除；非 AR 基线为 v0.5。
- `scripts/check_ckpt_load_cf5d68c.py`：Nano SFT 与 v0.6 checkpoint 在新框架下均可加载（missing 0、shape mismatch 0）。

## AR v0.1 实现（2026-09-27）

- 数据 `src/ar_dataset.py`、注意力 `src/ar_attention.py`、模型 `src/ar_model.py`、推理 `src/ar_inference.py`；细节见 `docs/ar_v0.1_design.md` 第 11 节。
- 框架补丁（默认关闭）：`supervise_temporal_causal_actions`、`gen_attention_override`，见 `packages/cosmos3/UPSTREAM.md`。
- normalizer v4（30Hz B3 增量）：`artifacts/cosmos3_action_contract/v4_frame_delta_30hz`。
- 训练输出：`outputs/joint_video_hand_pose/ar/<run>`，启动日志 `outputs/joint_video_hand_pose/ar/launch_logs/`。每步 1 个样本、CP1/FSDP8，8 卡 H800 显存峰值约 55 GiB。
- 测试：`tests/` 共 111 项（其中 4 项 GPU 测试在 CPU 上跳过）。

