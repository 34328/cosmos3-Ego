# Two-stage IT2V -> Joint 训练方案 v1

## 目标

先让 Nano SFT 在 EgoVerse 视频分布上学习纯首帧条件视频生成，再从该完整 DCP 初始化
joint video + hand-pose 训练。两个阶段使用同一批 `36 episodes / 181 segments`，不改变
Cosmos 模型架构，也不修改 `packages/cosmos3` 核心源码。

```text
Nano SFT DCP
  -> Stage 1: it2v_v1（只有 video target/loss）
  -> Stage 2: joint_from_it2v_v0（video + action target/loss）
```

## 两阶段共用的数据合同

- episodes/segments manifest、segment 起止帧、时序抽样索引完全相同。
- 原图 `640x360`，送入 VAE 前只在底部 reflection-pad 到 `640x368`，解码后裁回
  `640x360`。
- 帧数满足 `T=1+4n`；85K token cap 和 80/70/60/50% 同步抽帧策略不变。
- 使用训练和推理字节一致的 Cosmos structured JSON prompt：episode task context +
  segment instruction、真实 duration/FPS、`640x368` resolution、ego framing。
- 首帧是唯一视频 condition：`condition_frame_indexes_vision=[0]`；其余帧为 noisy
  future video target。
- 继续复用 Cosmos 原生 VAE、packing、FSDP/CP、flow matching 和 DCP 读写。

## Stage 1: `it2v_v1`

### 模型和 batch

初始化：

```text
/mnt/checkpoints/Cosmos3-Nano-dcp-sft/iter_000048464
```

Stage 1 使用 Cosmos 基础 `OmniMoTModel` 的视频训练路径，不使用项目的 visibility/action
loss adapter。每个 `SequencePlan` 必须满足：

```text
has_text   = true
has_vision = true
has_action = false
condition_frame_indexes_vision = [0]
```

batch 中不得出现 `action`、`domain_id`、`raw_action_dim`、`hand_visibility`。因此没有
action token、action noise、action prediction loss，也不存在通过 `action_loss_weight=0`
伪装纯视频训练的情况。

模型配置仍保留 `action_gen=True`，但 optimizer 只选择 Cosmos 官方 vision SFT 的四组参数：

```text
moe_gen / time_embedder / vae2llm / llm2vae
```

这是相对官方 `vision_sft_nano` 唯一必要的 checkpoint 差异。官方 recipe 会设置
`action_gen=False` 并从 checkpoint 中移除 action 分支；本项目第二阶段还需要原始 Nano SFT
action 权重，所以必须保留完整 action 分支，但冻结
`action2llm / llm2action / action_modality_embed`。Stage 1 保存前后这些权重必须 bitwise
一致。

### Loss 和首轮运行

- 目标函数只有 Cosmos 原生 `flow_matching_loss_vision`。
- 单项 loss 不需要模态平衡，`loss_scale=1`；W&B 每 step 记录
  `loss/video_raw`、`loss/total`、`optim/lr_base`。
- 首轮建议保持 overfit_v0.0 的 base LR `2e-5`，先以 IT2V 当前运行配置完成 600 steps；
  在 300/600 保存完整 DCP 并立即运行固定样本推理。
- 输出目录：

```text
outputs/egoverse_it2v/train/v1/
outputs/egoverse_it2v/inference/v1/iter_*/
```

Stage 1 replay 只比较 generated video 与 GT video，不叠加未训练的 action；顶部保留原始
structured JSON 中的核心指令，视频统一保存为 H.264。

## Stage 2: `joint_from_it2v_v0`

初始化 Stage 1 最终模型：

```text
outputs/egoverse_it2v/train/v1/checkpoints/iter_000000600/model
```

`checkpoint.load_training_state=false`，只加载完整模型参数；optimizer、scheduler 和 iteration
从 0 重新开始。严格加载全部 `net.*`，不加载 EMA。

Stage 2 恢复 overfit_v0.0 已验证合同：

- WAM batch 同时包含 video/action；首帧 video 和首帧真实 57D state 为 condition。
- 使用 `cosmos3_action_contract/v2` normalizer 和冻结的 MLP-AE-15。
- 8 个 action 子块等权，visibility 规则不变，`lambda_out_of_fov=0`。
- video/action 权重 `10:7`；base LR `2e-5`，三个 action projection 使用官方 `5x` LR。
- 训练 `moe_gen / time_embedder / vae2llm / llm2vae` 和三个 action projection；理解路径、
  VAE、MLP-AE 继续冻结。
- 首轮同样建议 600 steps，在 300/600 保存 DCP 并立即生成现有两行两列 H.264 replay。

输出目录：

```text
outputs/joint_video_hand_pose/train/joint_from_it2v_v0/
outputs/joint_video_hand_pose/inference/joint_from_it2v_v0/iter_*/
```

## 实现边界

纯视频代码独立放置：

```text
cosmos3_egoverse_it2v/src/
cosmos3_egoverse_it2v/configs/
cosmos3_egoverse_it2v/scripts/
cosmos3_egoverse_it2v/tests/
```

只新增一个 EgoVerse video-only adapter、一个 Cosmos experiment config、启动/推理脚本和必要
测试；现有 overfit_v0.0 数据/action 代码及 `packages/cosmos3` 不改。`joint_from_it2v_v0`
只新增从 IT2V DCP 初始化的配置，不复制当前实现。

## 启动门槛

1. 同一 segment 在 IT2V 与 joint adapter 中的 RGB 帧索引、FPS、prompt、canvas 完全一致。
2. IT2V batch 无 action stream，只有四组 vision 参数得到有限梯度；冻结 action 权重训练前后
   bitwise 一致。
3. 8-rank CP=1/FSDP=8 完成真实 packed batch forward/backward/optimizer step，无 OOM/NaN。
4. IT2V DCP 能由 `joint_v7` strict-load，且 action 分支与原始 Nano SFT 权重一致。
5. 先比较 Nano SFT zeroshot、IT2V step 300、IT2V step 600 的同 seed 视频；通过后才启动
   joint_v7，避免把两个阶段的问题混在一起。
