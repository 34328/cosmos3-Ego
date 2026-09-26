# EgoVerse x Cosmos 3：TODO-2 掌心 Visibility 与 Pose Loss

> 版本：v0.6（原始图像域离线 visibility 合同）  
> 日期：2026-08-14  
> 状态：方案已实现并通过 loss 单元测试。  
> 范围：不改 Generator 架构，不改原生 57D action。

## 0. 核心结论

### 必须始终遵守的三条 Loss 规则

```text
1. visibility_gt 只能由 GT 掌心、GT camera 和 intrinsics 预先计算，必须停止梯度；
   严禁使用预测 pose 或预测 visibility 决定 loss weight。

2. 分子和分母必须使用同一个有效权重：
   L_side = sum(n * w * pose_error) / max(sum(n * w), eps)
   w = visibility_gt + lambda_out_of_fov * (1 - visibility_gt)

3. lambda_out_of_fov=0 且某侧所有 future frame 都不可见时，跳过该侧 group，
   并对剩余 camera/hand group 重新归一化；不能把该侧记成伪零 loss。
```

这三条同时成立时，手动调整 `lambda_out_of_fov in [0,1]` 只改变画外 pose 的相对监督强度，不会因为有效样本数量变化而系统性压低 hand loss，也不会给模型提供“把手预测到画外来逃避 loss”的训练捷径。

当前版本采用以下合同：

```text
不考虑画面内遮挡
Mecka hand pose 视为 GT
掌心中心在有效画面内 -> visibility_gt = 1
掌心中心在有效画面外 -> visibility_gt = 0
```

`visibility_gt` 只控制该手未来 pose loss 的相对权重：

```text
w = 1.0                     if visibility_gt = 1
w = lambda_out_of_fov       if visibility_gt = 0
lambda_out_of_fov in [0, 1]
default: 0
```

`lambda_out_of_fov` 是可配置超参数。overfit_v0.0 默认使用 `0`，即画外手不计入 pose loss；保留 `0.2` 和 `1.0` 作为后续对照实验。

视频 loss 始终完整计算，不受 hand visibility 影响。

## 1. Visibility GT 的计算

### 1.1 掌心中心

训练数据直接读取每帧、每只手的 `obs_ee_pose[t, :3]` 作为掌心中心：

```text
palm_world[t, side] = obs_ee_pose[t, side, :3]
```

当前 Mecka 数据审计确认 `obs_ee_pose` 的语义为掌心中心位置加 wrist rotation：

```text
obs_ee_pose[:3] == palm_world
obs_ee_pose[3:7] == obs_wrist_pose[3:7]
```

其中 `palm_world` 的独立审计等价于：

```text
mean(obs_keypoints[[0, 5, 9, 13, 17]], axis=0)
```

该 keypoint 平均只用于数据一致性检查，不作为训练时 visibility 的主数据源。`obs_ee_pose` 不增加 action 维度，也不作为额外模型输入；手部 21 点重建仍使用 `obs_keypoints` 和 wrist-local codec。

### 1.2 投影判据

visibility 在原始数据图像域离线计算，不依赖后续 Cosmos resize、padding 或 VAE crop：

```text
palm_cam = T_cam_from_world[t] @ palm_world[t, side]
(u, v)   = project(P_source, palm_cam)

visibility_gt[t, side] =
    finite(palm_cam, u, v)
    and z_palm_cam > 0
    and 0 <= u < 640
    and 0 <= v < 360
```

`P_source` 是当前 episode 的原始内参，实际读取路径为 `group.attrs["intrinsics"]["front_1"]`，有效域严格对应原始 `images.front_1: 640x360`。模型不 resize，只在底部 padding 8 px 得到 `640x368`，因此内参和投影坐标不变；新增 padding 不扩展 visibility 的有效域。visibility 始终在原始 `640x360` 中离线判断。

每个 episode 离线新增：

```text
left.obs_palm_in_fov_front_1:  uint8/bool [total_frames]
right.obs_palm_in_fov_front_1: uint8/bool [total_frames]
```

离线任务必须断言长度等于 Zarr `total_frames`、值域仅为 `{0,1}`，并记录计算版本、原始图像尺寸和所用 intrinsics。训练 loader 只按窗口的 `[window_start,window_end)` 同步切片这两个字段，映射为 `visibility_gt[T,2]`；不得再次做投影计算。

掌心在画外但仍有少量指尖留在边缘时，按 `visibility_gt=0` 处理；掌心在画内但部分手指出画时，按 `visibility_gt=1` 处理。

### 1.3 数据比例

Mecka 100h manifest segment 在原始 `640x360` 图像域的全量统计：

```text
frames                         10,700,531
left palm visibility=1        99.9137%
right palm visibility=1       99.8657%
left palm visibility=0        9,231 frames
right palm visibility=0       14,369 frames
```

画外帧比例很低，不会主导训练分布。

## 2. Pose Loss 的正确归一化

原生 57D action 保持：

```text
camera       [0:9]      9D
right hand   [9:33]    24D
left hand    [33:57]   24D
padding      [57:64]    7D
raw_action_dim = 57
```

对 flow velocity squared error，先分别对通道求平均：

```text
e_cam[t]   = mean((v_pred[t, 0:9]   - v_gt[t, 0:9])^2)
e_right[t] = mean((v_pred[t, 9:33]  - v_gt[t, 9:33])^2)
e_left[t]  = mean((v_pred[t, 33:57] - v_gt[t, 33:57])^2)
```

定义（`r[t]` 是 Cosmos 原生 `train_time_weight`；当前配置为 `uniform`，即恒为 1）：

```text
n[t] = 1 - condition_mask_action[t]
w_side[t] = visibility_gt[t,side]
              + lambda_out_of_fov * (1 - visibility_gt[t,side])
r[t] = train_time_weight(sigma[t])
```

对每个 segment `i` 单独计算逐组加权平均。visibility/noisy 权重进入分子和分母；`r[t]` 按 Cosmos 原生 flow loss 只进入分子：

```text
L_cam_i = sum_t(r * n * e_cam) / max(sum_t(n), 1)

L_right_i = sum_t(r * n * w_right * e_right)
            / max(sum_t(n * w_right), eps)

L_left_i  = sum_t(r * n * w_left * e_left)
            / max(sum_t(n * w_left), eps)

G_i = {camera}
      + {right if sum_t(n * w_right) > 0}
      + {left  if sum_t(n * w_left)  > 0}

d_camera = 9; d_right = 24; d_left = 24
L_action_i = sum(d_g * L_g_i for g in G_i) / sum(d_g for g in G_i)
L_action_local = mean_i(L_action_i)
```

当某个 segment 的左右手 group 都有效时，上式退化为 `(9*L_cam_i + 24*L_right_i + 24*L_left_i)/57`；只有 `lambda_out_of_fov=0` 且某侧在该 segment 没有任何可见 future frame 时，才从该 segment 的 `G_i` 中跳过该侧。

这样做的结果：

- `lambda_out_of_fov=0`：每个 segment 内只对 `visibility_gt=1` 的未来帧求平均，不会因为 mask 了一些帧而把整只手的平均 loss 人为降低；
- `0 < lambda_out_of_fov < 1`：画外 GT 仍参与训练，但相对画内 GT 降权；分母同步使用 GT 权重，保持 hand loss 的总体尺度稳定；
- `lambda_out_of_fov=1`：画内、画外 pose 等权。

rank 内 packed segments 等权平均。跨 rank 启用 Cosmos 的 `sample_level_loss_averaging=True`：框架以 `group_size * num_local_samples / global_num_samples` 缩放整个 video+action loss，配合 FSDP/DDP 的梯度平均后得到全局 segment mean。这样不会因为某个 rank 恰好 pack 了更多短 segment 而改变它们的全局权重，也不会改变 video/action 的相对系数。

如果 `lambda_out_of_fov=0` 且某只手在某个 segment 的全部 future frame 都不可见，只在该 segment 中跳过这一侧，并按该 segment 实际存在的 group 重新归一化；其他 segment 的同侧监督不受影响。segment 等权是与 Cosmos 原生 per-instance loss 一致的合同，帧数较长的 segment 不应仅因帧更多就获得更大的 sample 权重。

首帧 `a0` 是 clean condition，`n[0]=0`，不计算 future action flow loss。

## 3. 为什么不会诱导模型生成“画面里没有手”

关键约束是：

```text
visibility_gt = f(GT palm pose, GT camera pose, intrinsics)
```

loss 权重来自 GT，并且不参与梯度。它不能由预测 pose、预测 camera 或模型预测的 visibility 决定。

因此：

- GT 手在画内时，即使模型把手预测到画外，该帧仍使用权重 `1.0`，模型无法降低 loss；
- GT 手在画外时，权重由数据预先确定，模型不能通过改变自己的预测进一步降低权重；
- 视频 loss 永远使用完整 GT 视频，生成视频少画一只本应出现的手不会得到任何奖励；
- 数据中超过 `99.8%` 的逐手帧是 `visibility_gt=1`，训练分布本身也不会鼓励无手视频。

真正会产生退化捷径的错误写法是：

```text
错误：w = f(predicted visibility or predicted pose)
```

模型可以把预测 visibility 压到 0 来逃避 pose loss。本方案明确禁止这种 self-gating。

## 4. 不增加 Visibility 输出通道

保持：

```text
raw_action_dim = 57
[57:64] = padding 0
```

不使用 `[57:59]` 预测 visibility，因为它可以由预测 camera、预测 hand keypoint 和 intrinsics 确定性计算。

训练期使用 GT 掌心计算 `visibility_gt`；推理期从预测的 21 个 keypoint 计算掌心，再投影得到预测结果对应的 `hand_in_fov` metadata。

首帧输入合同不变：任务提供真实的左右手 pose 时，无论掌心是否在 RGB 内，都作为 `a0` clean condition，不填零或 missing sentinel。

## 5. 已实现接口与验证

当前项目侧实现包含：

```text
读取 GT 掌心并完成投影
visibility_gt: BoolTensor[T, 2]
可配置 lambda_out_of_fov
逐手加权平均 action flow loss
画内/画外分组指标
```

单元测试必须持续验证：

1. 离线掌心投影 overlay 与原始 `640x360 images.front_1` 一致，字段长度和值域通过断言；
2. `lambda=0` 时实现结果等于手算的 visible-only mean；
3. `lambda=0.2` 时画外误差相对画内误差为 `0.2`，但 hand group 总体尺度不随可见帧数量系统性降低；
4. 修改某个预测 pose 不会改变同一 batch 已计算出的 `visibility_gt` 或 loss weight；
5. camera loss、video loss、另一只手 loss 和 `[57:64]` padding 不受该侧 visibility 影响。
6. 不同长度 segment 先各自归一化再等权平均，不能按 future frame 数加权；返回的 per-sample loss 数量必须等于本地 segment 数。
7. 某侧在一个 segment 全部不可见时只跳过该 segment 的对应 group，不能影响 packed batch 中其他 segment。

保留 `lambda_out_of_fov = 0 / 0.2 / 1.0` 的配置能力，overfit_v0.0 基线使用 `0`。
