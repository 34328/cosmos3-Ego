# 无首帧 Action State 的 T-1 Action 合同 v1

> 状态：设计稿，尚未实现。后续与正式大数据 action contract 一并修改；不回改 v0.5/v0.6
> checkpoint 及其 replay。

## 目标

对齐 Cosmos 官方 hand-pose 的时间布局：长度为 (T) 的视频只包含 (T-1) 个
action token，不再把首帧状态 (S_0) 放进 MoT。

```text
video:   I0        I1        I2       ... I(T-1)
action:       A0->1     A1->2     ... A(T-2)->(T-1)
```

每个 action token 仍为 57D：

- camera/wrist：`inv(T_t) @ T_(t+1)` 的 frame-to-frame pose9；
- hand：目标帧 (t+1) 的 20 个非腕点 wrist-local 60D，经冻结 AE 得到的 15D
  latent；hand latent **不做时间差分**。

首帧 RGB/VAE token 是唯一的模型 condition。首帧 pose 只作为离线解码绝对轨迹的
anchor，不再作为 action token 输入模型。

## 必须同步修改

1. **Action builder**
   - 训练输出从 `[T,57]` 改为 `[T-1,57]`；
   - pose 全部使用 future frame-delta normalizer；
   - hand 输出使用 `z[1:]`，对应目标帧 `I1...I(T-1)`；
   - 新合同不加载、不记录 `state_normalizer`。

2. **Dataset 与 loss 对齐**
   - video 保持 `T` 帧；
   - `hand_visibility` 使用 `visibility[1:]`，与目标帧 hand state 对齐；
   - WAM 下 action condition mask 应全为 0，第一条 transition 也参与 loss；
   - 57D→64D padding、8 个 action 子块及 loss 权重保持不变。

3. **Packing 与位置编码**
   - 复用 Cosmos 原生 SequencePlan Case A：`action_length=video_length-1`；
   - 断言 `condition_frame_indexes_action=[]`；
   - 断言 `action_start_frame_offset=1`，使 action[0] 的 mRoPE 时间位置对齐
     `I1`；
   - 视频 VAE、FPS modulation、LR、loss 和 attention backend 不随本改动变化。

4. **Inference**
   - 不再要求 `initial_action57.json` 或 `--action-path`；
   - 初始化 `T-1` 个待采样 action token，全部为 noisy/generated；
   - 输入 JSON 的 `num_frames` 仍表示视频帧数 T，避免把视频长度误改为 T-1。

5. **Decode、replay 与 metrics**
   - 新增显式 initial anchor：F0 headcam、左右 wrist transform；若需绘制首帧手，
     另存首帧 wrist-local keypoints/latent；
   - 所有预测 pose 先用 future normalizer 反归一化，再从 anchor 积分：
     `T_(t+1)=T_t @ DeltaT_(t->t+1)`；
   - hand latent 直接解码为目标帧局部手型，再由积分后的目标帧 wrist transform
     转到 F0；
   - reference action 同样保存为 `[T-1,57]`；
   - metrics 不再跳过 `prediction[0]`，删除旧的
     `condition_slot_normalized_max_abs_error`；
   - replay 若展示第 0 帧，必须标明其 pose 来自 anchor，而非模型预测。

6. **Artifact 与兼容性**
   - 新建 action contract 版本，manifest 明确
     `has_initial_action_state=false`、`action_alignment=destination_frame`；
   - future normalizer 必须由正式 train-only frame-delta 数据重新统计；当前 36-episode
     B3 normalizer 仅保留给旧 overfit checkpoint；
   - metadata 按 contract version 分派 legacy/new decoder，旧结果禁止被新逻辑覆盖；
   - token 长度和 packing 发生变化，不能从旧 run 的 optimizer/dataloader state 续训。

暂停的 temporal action mask 若未来恢复，必须先适配 T-1 action 索引；不能继续假设
video/action 等长。

## 最小验收

- 数据断言：`len(video)=T`，`len(action)=len(visibility)=T-1`；
- 数值断言：action[0] 的 rigid pose 等于 `inv(T0)@T1`，hand 等于 frame 1 latent；
- SequencePlan 断言：无 clean action，mRoPE offset 为 1；
- round-trip：给定首帧 anchor，GT action 解码可恢复 frames 1...T-1；
- inference smoke：输出恰为 `[T-1,57]` 且第一条 transition 被采样；
- legacy regression：v0.5/v0.6 旧 metadata 仍走含 S0 的旧 decoder。
