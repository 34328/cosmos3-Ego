# AR v0.2：联合去噪、多样本 packing 与统一动作损失

- 日期：2026-09-27（2026-09-28 改为“逐块条件”方案）
- 状态：当前动作表示升级为 `fixed_camera_wrist_local_delta_latent_v1`，布局仍为每块 U_k＋S_k 条件和 V_k＋A_k 联合目标。代码与CPU回归已完成；PCA15已通过119个heldout episode的重建验收；两套57D统计已拟合，并通过119个heldout episode代表窗口的往返验证。真实GPU／WAM质量验收尚未完成。实现与实验状态见 [experiment.md](experiment.md)，本文定义当前方案与验收要求。
- 目标：**优先提高 action 质量；视频作为联合学习与推理的辅助模态，同时用于检查视频—动作一致性。** 不以视频观感改善代替动作指标。
- 依据：AR V0.1 实验的失配与漂移观察（AR V0.1／V0.2 历史文档保留）。本文已合入审阅建议及本轮修正；远端源码初查 `cebaf8a`，采样与 padding 复核 `9c28a2e`，仓库 `/mnt/lzh/cosmos-EgoWAM`，分支 `ar-video-action`。

**一句话**：一个逐块预测的 API。每块给块首图像 U_k 和块首 state S_k，模型同时输出该块的抽帧视频和 30Hz action。刚体平移与旋转的逐帧增量在固定块首相机轴中定义；手形是当前帧手腕局部坐标的编码差 Δz；训练和主评测条件来自 GT，部署时来自真实观测。

## 1. 本版范围与保留项

| 本版改动 | 要解决的问题 | 交付结果 |
|---|---|---|
| 训练、推理都改为 chunk 内联合去噪 | action 等待完整视频生成；训练读真实当前视频而推理读生成视频 | 当前带噪视频与 action 双向交互，一个采样循环同时更新两者 |
| 每个 chunk 输入条件 U_k＋S_k | 模型需从增量历史自行推导当前位置；首观测会离开窗口；部署没有观测入口 | 每块新增两个干净条件：块首帧图像 latent U_k 和块首相机系 state S_k。训练与主评测取 GT，部署取真实观测。未来 action 仍为逐帧增量，从 S_k 积分 |
| 持久推理 KV cache | v0.1 每步重算整个前缀，chunk 延迟随 rollout 长度增长 | 每块先预填 U_k/S_k，当前块 30 步联合去噪，完成后一次 V/A 干净写入；保留前 15 个历史 chunk，有界缓存，延迟不随 rollout 长度增长 |
| 每卡多样本动态 packing | 每卡每步仅 1 个 clip，有效 batch 小 | 同一 packed forward 容纳多个隔离样本，正确计算全局样本平均 loss |
| 新表示专用归一化＋整体 action loss | 旧统计属于另一套坐标／手形语义，且八块等权引入额外权重 | state 57D 与 future 57D 分别用 train-only 数据重拟合并绑定 hash；目标为 `L_video + 1.0 × L_action` |

保留 Cosmos3-Nano 主干、冻结视频 VAE 和 57D action 容量；视频 stride=2、每 latent 帧 K=8 个 action、训练和推理均固定 C=4、clip T∈{33,65,129}。源帧索引和 30Hz action 不变；`speed_factor=0.5` 暂保留以控制变量。按2026-09-29用户决定，手形 codec 改为左右手独立 PCA15，取消 MLP codec 重训。只在当前744个 train episode 的有效窗口覆盖源帧上拟合，重叠帧去重；119个 heldout episode 仅验收，不参与拟合或选择维数。旧 v2_4 原样保留，不直接复用其权重。

57D 字段顺序不变：相机 9D＋右手腕 9D＋右手形 15D＋左手腕 9D＋左手形 15D。state 中保存块首刚体绝对状态和两手完整 latent；future 中保存相机／双腕的帧间增量以及两手 `Δz`。每只手的 20 个非手腕关键点经新 codec 压为 15D，**不是 20 个独立关节 loss，也不等于 20 DoF 控制表示**。64D 容量末尾 7D 始终为零，不参与归一化、加噪或 loss。

本版新训练统一从官方 `/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp` 初始化，新增类型 embedding 随新训练学习并保存。旧 V0.2 checkpoint、旧 codec 和旧归一化只作历史参考；它们属于不同表示，不能直接接到 `fixed_camera_wrist_local_delta_latent_v1` 上继续训练或推理。

本轮不引入 Motus2 action-first、value、self-forcing、机器人控制接口或额外高维手形 tokenizer，也不做少步数采样或蒸馏。**持久推理 KV cache 纳入本版必交付范围**，为机器人控制降低推理延迟；保留“截断到当前块末尾的完整因果前缀重算”作为正确性参考路径（第 6.1 节），正式缓存路径不得每步重算历史。本版要求延迟不随 rollout 长度增长，但不承诺实时。**条件 token 的 RoPE 时间修正（放在块首帧 b_k）纳入本版**，并在新基线中同样修正。

采样参数统一以第 2.2 节配置表为准，覆盖推理、回放、质量评测和完整采样 smoke；不设低步数快捷入口。训练随机时刻的 flow 目标和单步数值单测不展开完整生成链。

### 1.1 训练前先定位 v0.1 的失败来源

原计划作为训练前的第 0 步；当前尚未完成，保留为待补诊断（不宣称已通过）：对已有 v0.1 训练集／held-out 回放离线运行同一个手部姿态估计器，同时处理 GT 视频和生成视频，保留逐帧检测置信度、左右手匹配、遮挡与漏检记录。先在 GT 视频上用原始标注验证检测器可靠性；估计器不是生成视频上的真值，不能把检测失败当零误差。

在真实生成的采样帧、统一图像尺寸和时间戳上，分别记录：

1. **生成视频中的手 vs GT 视频／标注中的手**：检查视频轨迹是否偏离记录的未来，同时报告检测覆盖率和可见性变化。偏离 GT 不自动等于视频不合理；另做人工抽查。
2. **预测 action 的投影 vs 生成视频检测出的手**：检查 action 和画面是否一致。优先比较共同可见的 2D 关键点，报告像素误差及按手部尺度归一化的误差；不把单目估计器的未标定 3D 输出直接当毫米真值。
3. **预测 action vs GT action**：保留物理域位置、旋转和手形误差，不能用视频一致性替代任务指标。

采用第 3 节的 GT camera 投影口径，同时分别展示腕部位置与去腕部平移后的手形误差。相机轨迹偏差会影响 2D 重合，需单列 camera 误差并检查投影约定；不能将所有不重合归因于 action。冻结检测器版本、阈值和评测帧清单，对比时报告共同有效帧数及各自覆盖率，避免漏检掩盖失败。

诊断报告区分：视频偏离 GT 但 action 跟随视频、视频接近 GT 但 action 不一致、两者都异常、检测证据不足。混合情况按 clip／chunk 分析，不强行二分。它帮助确定后续优先级，**不能提前证明联合去噪一定有效**。已有 20 步回放可作为历史诊断素材，但不进入 V0.2 的 30 步公平对照，也不为诊断重新运行低步数采样。

### 1.2 v0.1 遗留问题的本版处理

| 问题与证据 | V0.2 处理与边界 |
|---|---|
| oracle 下动作较好，换生成视频后接近“不动”基线 | 第 0 步诊断＋当前 chunk 联合去噪；以实际动作精度和视频一致性验收 |
| 首 chunk 即使 oracle 也更差 | chunk 1 与其他块同构（U_1＝首帧，S_1＝首帧 state），修条件 RoPE；chunk 1 仍无运动历史，单列报告 |
| 逐帧增量积分累积误差 | 每块从 S_k 重新积分；块内刚体增量使用固定块首相机轴，手形使用腕局部编码差。自由生成时下一块刚体 state 换轴，手形 z 原值继承；预测增量的积分漂移另报 |
| GT 历史训练、生成历史推理的差异 | 新增 `pred_history` 诊断（GT 当前条件＋预测历史）量化差异；历史视频加噪（第 2.3 节）暂缓，出现具体问题再做 |
| 每卡单 clip、八块等权和尺度疑问 | 多样本 packing、整体 MSE；尺度校准保留为暂缓的候选实验 |
| 首状态离开窗口后没有显式状态；没有闭环观测入口 | 每块都有 U_k/S_k，且以块首相机为参考，数值有界、不依赖全局位姿；部署时由当前相机图像和手部追踪直接得到。真实机器人接入留待后续 |
| 一致性检查无失败阈值、smoke 未验收保存恢复 | 补齐真实训练／推理数值对照、50 个训练更新的可靠性检查和独立恢复入口；已有训练日志的有限梯度记录不能替代恢复测试 |
| GT-history 曾覆盖早期预测 | `cebaf8a` 代码已使用独立输出缓冲区，属已修复项；V0.2 重构时保留多 chunk 回归防退化 |
| 历史重算导致推理慢 | 本版实现持久 KV cache、窗口淘汰及逐步数值对照；固定 30 步，验收相同大小 chunk 的延迟不随 rollout 长度增长，不承诺实时 |
| 36 个训练 episode、4 个 held-out episode，且旧训练非官方初始化 | 固定清单、报告样本量与逐 clip 结果；新对照官方初始化。当前已扩为 744 个训练／119 个验证 episode（见 experiment.md）；旧小样本结果不作泛化结论 |

**解码与指标**：每块输出一律从该块的 S_k 积分（第 2.0 节）。
- **块内误差**是主指标，在块首相机系下计算：逐块末端腕部／相机的平移与旋转误差、手部 MPJPE、块内运动方向余弦，并与“不动”基线比较。
- **整段漂移**只在自由生成诊断中报告：S_(k+1) 取预测末态，全局轨迹由预测相机逐块左乘得到，途中不用 GT 重置。
- v0.1 手腕末端约 10–15 cm、相机最大约 19 cm，仅是原数据与原初始化下的历史观察。

## 2. 训练与推理联合去噪

### 2.0 每块条件 U_k＋S_k：本版采用的方案

**定义**：设 chunk k 的第一条未来 action 源帧为 b_k+1，b_k 为块首帧（相对 clip 首帧；读 episode 时加随机窗口起点）。
- 固定 C 时 `b_k=(k−1)×C×K`；实现用边界索引，训练固定 C=4，兼容末尾不满块。
- 例如 C=4、K=8：chunk 1 的 b_1=0，预测 action 1…32；chunk 2 的 b_2=32，预测 33…64。
- b_k 同时是上一块视频的最后一帧。

chunk k 由四部分组成：

| 部分 | 内容 | token | 噪声／loss |
|---|---|---|---|
| U_k | 源帧 b_k 的图像 latent（单帧经 VAE 编码为 1 个 latent） | 640×368 下 240 | 干净，不计 loss |
| S_k | 源帧 b_k 的 state（下述方案 B） | 1 | 干净，不计 loss |
| V_k | 源帧 b_k+2 … b_k+8C 的 C 个视频 latent | 240×C | 加噪，计 video loss |
| A_k | 源帧 b_k+1 … b_k+8C 的 C×K 条逐帧增量 action | C×K | 加噪，计 action loss |

没有单独的 chunk 0：chunk 1 的 U_1 就是首帧，S_1 就是首帧 state，各块结构相同。

**逐块 VAE 编码**：每块把 `[b_k, b_k+2, …, b_k+8C]` 共 1+4C 帧送入因果 VAE，得到 1+C 个 latent，首个即 U_k，其余为 V_k。
- 与 v0.1 的整段编码不同：V_k 的因果上下文从 b_k 开始，与部署时“拿当前帧开始预测”一致。
- 训练固定 C=4，按该块边界进行编码，不再随机选择 C。远端 VAE 本来就在线编码（`omni_mot_model.py` 的 `balance_vae_encode`），按块编码即可，不做预存缓存。
- 按块编码的开销要实测。v0.1 整段编码约 0.6 s。

**固定块首相机坐标轴**：令块首相机到世界的位姿为 `B=H_bk=(R_B,p_B)`。对块内任意相机或手腕刚体 `X_t=(R_t^W,p_t^W)`，先统一变换到 B：

`p_t^B = R_B^T (p_t^W-p_B)`，`R_t^B = R_B^T R_t^W`。

这里的上标 B 表示“数值分量始终沿块首相机轴表达”。此句仅指相机／双腕刚体，块内不切换到上一帧相机轴；手形单独采用当前帧腕部局部轴。

**S_k（57D 初始条件）**：
- 相机 9D：物理状态是 `(p=0,R=I)`，编码后固定为 0；模型入口断言这 9 维为 0。
- 双腕各 9D：保存块首腕在 B 中的完整平移和 6D 旋转，即 `p_h,bk^B、R_h,bk^B`。
- 双手形各 15D：`q_h,t = (R_h,t^B)^T (keypoints_h,t^B-p_h,t^B)`，即当前帧手腕作为原点和坐标轴的 20 个非腕关键点。`z_h,t=E(q_h,t)`，S_k 保存完整 `z_h,bk`。q 不含整只手的刚体朝向；不是块首手腕轴。

**A_k（57D future）及解码**：对 `t=b_k+1…b_k+32`，前一时刻记为 `t-1`。
- 相机和双腕的平移增量：`Δp_t=p_t^B-p_(t-1)^B`；解码用 `p_t^B=p_(t-1)^B+Δp_t`。
- 相机和双腕的旋转增量：`ΔR_t=R_t^B (R_(t-1)^B)^T`；解码用**左乘** `R_t^B=ΔR_t R_(t-1)^B`。9D 刚体字段保存 `Δp` 和 `ΔR` 的连续 6D 旋转表示。
- 双手形增量：先逐帧编码 `z_h,t=E(q_h,t)`，再保存 `Δz_h,t=z_h,t-z_h,t-1`；解码用 `z_h,t=z_h,t-1+Δz_h,t`，随后 `q_h,t=D(z_h,t)`。
- 关键点重建为 `keypoints_h,t^B=p_h,t^B+R_h,t^B D(z_h,t)`。腕旋转在此应用一次，腕位姿错误仍会影响最终关键点。

平移和旋转必须分别积分。禁止把 `(ΔR,Δp)` 拼成齐次增量后左乘旧位姿，因为那会错误地把旧平移也旋转。手形目标仍是完整标准化 latent 的差 `E(q_t)-E(q_t-1)`。对当前 PCA，该差等于 `(q_t-q_t-1)·Cᵀ / σ_z`；不能直接调用带均值中心化的 `E(q_t-q_t-1)`，否则会多减一次均值。

**换块／generated 重建**：设本块末相机在 B 中为 `C=(R_c,p_c)`。下一块坐标轴以 C 为基准：
- 腕部 `p'_h=R_c^T(p_h-p_c)`，`R'_h=R_c^T R_h`；下一块相机重置为 `(0,I)`。
- 手形 `z'_h=z_h` 原值继承，不调用 codec encode/decode，也不旋转 15D latent。手腕局部 q 在整个场景刚体换轴后不变。
- 此处消除的是换块反复 E/D 的误差来源，不消除预测 Δz 自身的积分误差。

每块只依赖 S_k 和 A_k 解码。`gt`／`pred_history` 的下一块 S 直接由 GT 边界观测构造；`generated` 才执行上述预测刚体末态换轴与手形 z 原值继承。全局轨迹如需展示，再用块首世界位姿恢复，不改变模型内部表示。

| 场景 | U_k、S_k 来源 | 历史 V/A |
|---|---|---|
| 训练（teacher forcing） | GT 帧 b_k | GT |
| 主评测 `gt` | GT 帧 b_k（验证集完整视频） | GT |
| 诊断 `oracle` | 同 `gt`；当前 V_k 也固定为 GT | GT |
| 诊断 `pred_history` | GT 帧 b_k | 模型生成：每块预测写入缓存 |
| 诊断 `generated` | U_(k+1)：生成 V_k 解码后取最后一帧，再单帧编码；S_(k+1)：预测末态按上述规则换到新相机轴，手形 z 原值继承 | 模型生成 |
| 部署 | 真实相机当前帧＋手部观测转到相机系 | 模型生成或真实观测，按部署决定 |

训练和主评测每块条件直接取 GT；`generated` 的图像 U 需解码末帧后单帧编码，手形 z 则直接继承；`pred_history` 条件仍取 GT，只把历史换成预测。每块保存 `condition_source、boundary_source_index、boundary_time`，供复现；这些是审计 metadata，不输入模型。

**模型入口**：
- S_k 补零到 64D，复用 action 输入投影，并加可学习的 `state/action` 类型 embedding。
- U_k 走视频输入路径，加 `condition/target` 类型 embedding 以区分块首条件帧和待预测视频。
- U_k、S_k 的 timestep 为 0，RoPE 放在 b_k 的模型时间。二者不加噪、不做 Euler 更新、不计 loss，但输入分支会从 V/A loss 获得梯度。
- 取消旧“首帧 state 重复 K=8 次”的 AR 布局；K=8 仍只规定未来 action 与视频 latent 的配比，非 AR 路径不变。重复 8 次是旧打包布局的实现选择，不是 flow matching 的要求。
- 布局版本 `joint_chunk_cond_v1`，替代 `joint_state_single_v1`。训练、重算推理、缓存推理和评测共同读取显式的角色／chunk／源帧索引，不凭 action 行数推断布局。
- 末尾不满块仍有一对 U_k/S_k，未来行只取真实存在的数据。

**Dataset／packer**：Dataset 提供原始逐帧 pose、3D 关键点和 RGB 帧；按固定 C=4 构造各块的 U_k/S_k 和 V_k/A_k。不能从旧手形 latent 或旧 action 行反推新表示。packer 显式携带 `token_role、sample_id、chunk_id、source_index、condition_mask、action_representation`；训练、推理和评测都要求 `action_representation=fixed_camera_wrist_local_delta_latent_v1`，禁止凭 action 行数猜布局。

**PCA codec 与归一化合同**：
- 输入为当前帧 wrist-local `q=R_wristᵀ(kp−p_wrist)`，20点展平为米制60D。先中心化，不做输入逐维标准差缩放；用FP64协方差特征分解取15个正交主成分，固定每个主成分符号。左右手各自记录 episode 清单、源窗口／原始数据 hash、seed=42 和 explained variance；不使用随机SVD。
- 设行向量 `x=flatten(q)`，`C` 为15×60主成分矩阵：`z=((x−μ_x)·Cᵀ−μ_z)/σ_z`；`decode(z)=(z·σ_z+μ_z)·C+μ_x`。拟合时 `μ_z=0`、`σ_z` 为对应特征值平方根，退化维数采用显式floor。60→15是有损投影，decode是PCA重建，不宣称恢复被丢弃的45维。
- 产物目录：`cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v3_wrist_local_pca15_train744/`。左右手为 `right_pca15.pt`／`left_pca15.pt`，附 manifest 与各自通过验收的 sidecar。runtime 按显式 architecture 分派 PCA／历史 MLP，57D表示版本不变，但codec hash必须严格绑定，禁止隐式换codec。
- checkpoint 与 held-out sidecar 绑定表示版本、INPUT_FRAME、左右手、权重 SHA、训练 episode 清单、数据及统计 hash。不能仅改元数据给旧权重换名。
- 左右手分别要求 held-out 重建 mean≤5mm、P95≤15mm；这些是当前工程门槛，不是官方标准。实际多块 reanchor 测试断言 z 不变、重建点经刚体变换前后物理一致，覆盖非零腕／相机旋转；不再使用会主动引入 E/D 的旧 rebase 门槛。仅重复解码同一个 z 不能代替该测试。
- `state_normalizer` 与 `future_normalizer` 都是独立的 57D train-only 统计。state 统计覆盖相机常量槽、双腕完整块首状态和完整 z；future 统计覆盖相机／双腕 `Δp,ΔR` 与双手 `Δz`。相机 state 前 9 维固定为 0，不从噪声估计尺度。
- 两套统计都只按正式 C=4 边界和训练窗口拟合；验证集只做往返与分布检查。统计文件必须绑定 `fixed_camera_wrist_local_delta_latent_v1`、冻结 codec 左右权重 hash、数据清单 hash、拟合种子和字段切片。任一项缺失或不匹配，训练、恢复和推理都显式失败。
- 旧 F0、旧局部 SE(3)、旧绝对手形 latent 或 C=1…4 混合统计保留为历史产物，不能自动迁移或冒充新统计。总训练目标仍只有 video loss 和 action loss。

### 2.1 注意力与条件

**统一单位**：chunk k＝{U_k, S_k, V_k, A_k}，是一个历史槽位。chunk 之间因果；当前 chunk 内带噪视频和 action 双向可见；U_k/S_k 是当前块的干净条件。

| Query | 可以读取 | 禁止读取 |
|---|---|---|
| 当前带噪 V_k / A_k | 本样本文本、窗口内干净历史 U/S/V/A、当前 U_k/S_k、当前带噪 V_k/A_k | 当前干净 V_k/A_k、未来 U/S/V/A、其他样本 |
| 干净条件流的 V_k/A_k | 本样本文本、窗口内更早干净 chunk、当前 U_k/S_k、同 chunk 干净 V/A | 任意带噪 token、未来 U/S/V/A、其他样本 |
| 干净 U_k / S_k | 本样本文本、同块 U_k/S_k | 历史与未来 chunk、当前 V/A（干净或带噪）、其他样本 |

联合 mask 用一个 chunk id 同时标记 U_k、S_k、V_k、A_k，不再使用 `V=2k、A=2k+1` 的模态先后关系。对 V/A query：clean→clean 允许 `k_key≤k_query`；noisy→clean 仅允许 `<`，**唯一同块例外是条件 key U_k/S_k**；noisy→noisy 仅允许 `=`。U_k/S_k 只编码文本和本块条件，历史融合交给 V/A query，避免条件 cache 吸收答案，也不随当前去噪变化。

**历史窗口长度固定为 15 个 chunk，当前预测 chunk 不计入历史窗口。** 配置统一命名为 `history_window_chunks=15`，下文简记 H=15；训练和推理使用相同口径，不再随机采样旧 block 窗口。对当前 chunk k，历史可见条件为 `1≤k−j≤H`，即最多读取 `k−15…k−1`；序列开头不足 15 个时只读取已有历史。当前带噪 V_k/A_k 的双向交互、clean refresh 的同块交互另行允许，不占历史额度。训练 C 固定为 4，不进行 chunk 长度随机采样。配置和 mask 版本一并更新，禁止将旧窗口字段直接解释为 chunk 数。

继续用两次前向实现 teacher forcing：先按上述角色 mask 一次算全部 U/S/V/A 干净 K/V，再算全部带噪 V/A 目标；保留干净 K/V 的梯度。U/S 不在 noisy 流中重复。干净流中的 GT 当前 V/A **不得被当前 noisy 块读取**。**U_(k+1)/S_(k+1) 就是当前块的末帧画面和末态**，虽是条件 token，也必须对 chunk k 屏蔽，这由 chunk 因果自然保证，需单测。文本只属于本样本，不能吸收 GEN 内容再回流。

**三个 mask／有效性概念互不替代**，第 5.2 节的 loss mask 不能替代输入检查：

| 概念 | 规则 |
|---|---|
| loss mask M | 新 wrist-local 路径采用方案 A：通过跟踪审计的全部真实 future 行、全部 57D 参与监督，不按 palm-in-FOV 屏蔽；仅排除干净条件、末尾 7D 和结构 padding。跟踪无效整窗拒绝。legacy 路径保留原 FOV 屏蔽 |
| 输入有效性 | 模型接收 64D，末尾 7D 在初始化、加噪、每次 Euler 更新和 clean refresh 时恒为 0；有效行必须为有限值 |
| attention mask | 按 chunk 因果、15 个历史 chunk、样本隔离和结构 padding 控制；padding 行不可作有效 key，不按手部 FOV／监督 mask 屏蔽整个 action token |

不可见不等于坐标缺失：视野外但仍有可靠、有限跟踪坐标的手部，沿用原值进入模型并参与监督，不新增可见性输入。真正缺失、NaN/Inf、非法旋转或已确认损坏的标签，不原样进入 noisy 输入或历史 cache；现有 `invalid_frames` 在构建 action 前排除受影响的整个窗口并记录原因／数量，边界帧和相邻源帧同样检查，不改成逐行 loss 屏蔽或隐式补值。新表示保留原始 `hand_visibility` 字段供审计与可选的可见／不可见分组评测，但其值不参与 loss 分子、分母或权重；不能把未来 GT 可见性作为模型特征或推理条件。不声称这已消除 GT／generated 历史分布差异。上述方案 A 仅适用于 `fixed_camera_wrist_local_delta_latent_v1`，legacy 行为不变。

**边界缺标的明确处理**：S_k 需要边界帧的相机、双腕和手形，边界相机、任一手腕或该手 codec 所需关键点真正缺失时，该候选窗口不可进入训练；不把缺失值置零，不从未来插值，也不新增左右手有效位。视野外但可靠跟踪的手部不按缺标排除。实施时按固定 C=4、各 clip 长度和各 episode 输出 train／held-out 数据审计：候选窗口数、原 action 输入有效性规则排除数、新增 state 边界规则的额外排除数、相机／左手／右手及原因分布、最终保留数。边界本来就在原始动作序列中，不能未统计就声称新增 state 必然显著增加排除率。

拟合 state 统计前冻结**有效窗口清单**，之后训练、评测及可能的后续对照都用这一份。训练前若某种计划的 C 无有效样本，或 held-out 清单不能覆盖原定各 episode，先解决数据覆盖。评测遇到 GT 边界缺标时跳过该项并报告覆盖率。有效位＋显式缺失编码仅在审计证明有需要后作为独立数据适配方案，不默默改变本版输入语义。

输入构造保留现有训练 `raw_action_dim` 清零和推理 `_zero_padding`，模型入口断言末尾 7D 为零。测试分别验证“构造阶段清零成功”和“绕过构造直接给入口非零 padding 时失败”，避免自动清零掩盖入口错误。

这取消了 action 必须以完整当前视频为条件的串行生成方式，使当前 V/A 共同演化；**不代表消除了训练与生成结果的分布差异**。训练的视频 σ 接近 0 时，action 仍会读取接近 GT 的视频，而推理读取的是生成结果；部署时历史为模型生成，与训练的 GT 历史仍有差异（第 2.3 节）。

### 2.2 加噪与采样

训练每个样本、每个未来 chunk 独立采样 `(σ_video, σ_action)`，各模态内部同 chunk 共用 σ，两份高斯噪声独立。保留 v0.1 的噪声分布和 shift 配置，包括 `shift_action=5`；必须分别传入视频、动作的 timestep，不能把“共同更新”写成“共用视频 timestep”。

对视频 latent 或校准后的 action z，均使用 `x_σ=(1−σ)x+σε`，目标为 `v*=ε−x`。训练计算随机噪声时刻的去噪目标，不展开完整采样链。

**本版采样配置（入口校验，并完整写入结果元数据）**：

| 项目 | 固定值／规则 |
|---|---|
| `video_steps` / `action_steps` | 30 / 30，联合同步更新，无提前结束；video-first 对照为串行 30＋30 |
| CFG | `guidance=1`，每步只做条件前向，不跑无条件分支；训练保留 `cfg_dropout_rate=0.1`，评测输入 dropout=0 |
| `video_shift` / `action_shift` | 5 / 5；640×368 当前视频实现取 `shift["480"]`，新配置显式保存数值 5，不再隐式查 key |
| sigma 序列 | `u_i=1−i/30`，i=0…30；`σ_i=s·u_i/(1+(s−1)·u_i)`，与 `flow_sigmas()` 一致 |
| timestep | 视频／action 分别传 `t=σ×num_train_timesteps`，两路当前均为 1000；训练仍独立采样各自 σ |
| Euler 更新 | `x ← x+(σ_{i+1}−σ_i)·v_pred`，两路预测来自更新前同一状态 |

当前两路推理 schedule 的数值相同，但接口必须独立传递，不能把训练独立 action timestep 覆盖为视频 timestep。独立 sigma 的数值单测不改变正式采样配置。U_k/S_k 不加噪、不更新、不计 loss。

推理使用持久 KV cache，每个 chunk 固定用 **30 次联合更新**：

1. **文本预填**：episode 开始时对本样本文本前向一次并常驻，不计入窗口。
2. **条件预填、初始化当前块**：为 chunk k 取得 U_k/S_k（来源见第 2.0 节表），对这 241 个 token 做一次零噪声前向，保存各层 K/V 及角色、绝对时间、chunk id；它们只读文本和彼此，不改旧 cache。初始化当前 V_k、A_k 的独立噪声。
3. **当前块联合前向**：只对当前带噪 `(V_k,A_k,σ^v,σ^a)` 前向，读取历史 cache 和当前 U_k/S_k，同时得到 `(v_video,v_action)`。本步的带噪 K/V 是临时数据，不追加进持久 cache，不跨步复用。
4. **同步更新 30 次**：两路各用自己的 `Δσ` 做 Euler 更新，预测来自更新前的同一状态，直至两路 σ 都为 0。
5. **保存预测、一次干净写入**：最终 V_k/A_k 先存入独立输出缓冲，所有指标从该缓冲读取。然后以零噪声对当前 V/A 前向一次并追加 K/V：`generated` 与 `pred_history` 写预测值，`gt` 写 GT 值。`gt` 不先写预测再覆盖，也不把 GT 写进预测输出。
6. **输出和推进窗口**：从 S_k 解码本块 action 并返回。`generated` 模式把末帧解码为 S_(k+1)，并把生成视频解码后取末帧重编码为 U_(k+1)；这两步在 action 返回之后，计入下一块的条件准备耗时。按窗口淘汰过期 chunk，继续下一块。

每块为 **1 次条件预填＋30 次联合去噪＋1 次 V/A clean refresh，共 32 次前向**，首块另加一次文本预填。预填和写入都不是采样步，耗时分开计时。

**缓存可见性与淘汰**：chunk j 的 U_j/S_j/V_j/A_j 同属一个槽位，H=15。
- 当前 k 读取历史 `max(1,k−15)≤j≤k−1`，外加当前 U_k/S_k；clean refresh 另读当前 V/A。当前块不占历史槽位。
- k=16 读取 chunk 1–15；k=17 时 chunk 1 整体（U_1/S_1/V_1/A_1）首次淘汰。
- 文本常驻。没有永久条件 token，每块由新的 U_k/S_k 携带当前观测。

每层只缓存未过期的干净历史 K/V，当前块另用临时工作区；用有界 ring buffer／等价布局，禁止持续拼接完整前缀。写入后按下一块需求淘汰，固定额外一个当前块的空间上界。RoPE 保留原始绝对时间和 chunk id，物理槽位复用不得将位置重置为 0。第一轮淘汰时必须逐 token 比较 key 可见集合，而不仅检查缓存长度；末尾不满块只写入真实 token，padding 不可被读取。

为满足延迟不随 chunk 序号增长，完整块的推理工作区按固定 C、分辨率、文本长度和 H=15 预分配，即 15 个历史槽位加当前块工作区；未填满窗口的槽位用无效 mask 占位，避免随前缀增长重编译、扩容或增大前向形状。kernel 若采用变长 key 布局，仍需证明窗口填满前后的延迟符合第 6.2 节门槛，不能只验收填满后的平台期。末尾不满块单列，不混入完整块延迟趋势。

常规联合生成不读取当前块的干净 V_k。另设 `oracle` 条件诊断：历史用 GT，当前视频固定为 GT、σ_video=0，只更新 action 30 步，之后按 `gt` 规则一次写入历史。当前视频 token 在联合注意力中读取变化中的带噪 action，其各层特征每步都会变，所以每步仍对当前 V/A 一起前向，只丢弃视频输出、不做视频 Euler 更新；不能把当前视频 K/V 算一次后缓存 30 步。它改变了 joint 的条件分布，**不是严格的性能上限**，不进入联合生成主结果或延迟验收，输出明确标注视频为 GT。

相同 30 步预算下，video-first 为“30 次视频＋30 次动作”，joint 为“30 次联合更新”；持久 cache 进一步移除每步历史重算。旧 v0.1 的 20＋20 是另一采样预算，不能用其耗时直接推导缓存的独立收益。**实际加速与质量必须测量，不承诺 2× 或实时**。正式性能测量运行完整 30 步，不以几步计时外推替代完整测量。

### 2.3 暂缓的可选实验：历史视频条件加噪

联合去噪改变当前块条件，历史仍存在 teacher forcing 差异。本版暂缓：完整 V0.2 中 `history_video_noise_prob=0`。若 `pred_history` 相对 `gt` 明显变差，再单独训练一组 p=0.5，其余配方与完整 V0.2 相同，按 `pred_history`／`generated` 的动作指标、漂移和视频一致性决定是否采用。以下为届时的实现要求。

实现为对条件流中的历史视频 latent 加噪：每样本独立决定是否启用，每个历史 chunk 共用一个条件噪声时刻；所有 U_k/S_k 保持干净，历史 action 暂不加噪。不得改变监督目标或把当前／未来 GT 泄漏给 noisy query。复用条件流时，某块带噪条件 K/V 只能被后续块读取；不能为不同 query 重复随机扰动同一份缓存。条件流必须携带正确的噪声 timestep，并保留干净目标副本；启用时它不再是全零 timestep 的“干净流”。需检查条件流其他 token 通过注意力受到的间接影响。此实验不扰动 U_k/S_k，不覆盖“训练用 GT 条件、自由生成用预测条件”的差异。

LingBot 参考源码 `wan_va/train.py:168–236` 对视频条件流以 0.5 概率加噪、action 条件流概率为 0，并传递 `cond_timesteps`；其采样边界 0.5–1.0 属于 scheduler 的 timestep 索引比例，**不能直接写成本项目 flow σ∈[0.5,1]**。V0.2 的条件噪声范围需在训练前依据本地 scheduler 映射显式确定并记录；不把此适配版本声称为逐项复刻 LingBot。原 video-first mask 还使该扰动影响 action 读取的当前视频，因此也不能把原机制解释为只扰动历史。

该消融只近似增强对历史视频误差的稳健性，高斯噪声不等于模型生成误差，不保证解决积分漂移。验证 p=0 路径与基线一致、U_k/S_k 不受扰动、跨样本隔离、未来不可见、噪声 timestep 正确，以及 GT／generated 两种历史下的质量变化。

## 3. 时间对齐与双栏回放

### 3.0 训练覆盖与长 rollout 边界

T 指抽帧后送入 VAE 的 RGB 帧数。正式 C=4 时，T=33／65／129 分别对应 2／4／8 个未来 chunk，每个完整 chunk 为 4 个视频 latent 时间组和 32 条 action。现有最长训练片段只有 8 个未来 chunk，尚未覆盖 H=15 填满后的训练分布。

正式推理固定 C=4：从第 9 块起，目标绝对时间超出当前训练片段的范围；第 17 块首次淘汰，C=4 训练未覆盖该边界。本版主要质量指标限于 chunk 1–8；至少 48 块的长 rollout 仅作缓存正确性／性能验收，不以其速度或数值一致性证明长时动作质量。更长训练 clip（例如 T=257）需先核对 episode 长度分布，留作后续数据方案，H 保持 15。

首次淘汰必须按正式 C=4 构造超出训练 clip 长度的同输入 cache／重算测试。该测试只证明实现等价和缓存有界，不能证明模型学会了该长时分布。历史 C=1…4 矩阵只保留在 experiment.md 的旧实现记录中，不作为新表示的训练或验收配置。

### 3.1 抽帧与速度分别计算

| 量 | 当前取值 | 含义 |
|---|---|---|
| 原始视频／action | 30fps／30Hz | 原始真实时间 |
| 视频 stride=2 后 | 15 个视频帧／真实秒 | action 仍保留 30 个／真实秒 |
| VAE 时间压缩 | 4 个视频帧 → 1 个 latent 时间组 | 同组配 8 个 action |
| speed_factor=0.5 后的模型标签 | 视频 7.5fps、action 15Hz | 两种模态共同把时间轴拉长 2 倍 |

**stride=2 本身不使运动变慢。** 抽帧后按 15fps 播放是原速，按 7.5fps 播放是半速；不能再乘一次 1/2 算成四分之一速度。若给模型的视频标签真是 15fps、action 是 30Hz，则对应 speed_factor=1。v0.1 实际默认标签为 7.5／15。

例如 chunk 1 的第一组未来数据：视频保留源帧 `[2,4,6,8]`（源帧 0 是 U_1），action 保留 `[1,2,…,8]`；第 j 组 action 为 `8(j−1)+1…8j`。C=4 对应 16 个视频帧、32 个 action，覆盖真实时间 `32/30≈1.07s`，模型时间 `32/15≈2.13s`。两者按时间段和各自 RoPE 对应，不要求 token 数相等。

U_k 和 S_k 的 RoPE 都放在源帧 b_k 的模型时间（S_k 用 `b_k/fps_action`，U_k 用对应视频时间）；未来 action 从 b_k+1 递增，条件 token 不推进 action 行号或视频时间。此布局替代首状态重复 8 次的旧方案，需要新训练。

### 3.2 默认回放：左右同速、同一时间轴

- **左侧**：原始 GT 视频，保留 30fps 连续画面；可叠加绿色 GT handpose。
- **右侧**：模型预测的抽帧视频＋红色预测 handpose；不要求补生成被抽掉的 RGB 帧。
- **默认 `real_time` 输出 30fps**：左侧每帧更新；右侧 15fps 预测画面按时间戳重复显示，handpose 使用该原始时刻的全量 30Hz 预测，每帧更新。视频缺失时刻的背景是保持帧，不是新生成画面；不做视觉插帧或动作平滑来掩盖误差。
- **可选 `model_time` 输出 15fps**：使用同样的合成帧序列，左右一起半速播放。不能左侧原速、右侧半速，却声称逐时刻对齐。
- 两栏标注源时间、chunk 编号、播放倍率。保存源帧索引、两路 fps 标签与视频帧→action 行的映射；U_k/S_k 作为条件 metadata 单独保存。旧 v0.1 回放走旧兼容路径。

**手部投影**（`ar_overlay` 改为逐块积分）：对 chunk k 内时刻 t，
1. 按第 2.0 节从 S_k 得到块首相机系下的腕部位置 `p_h^B(t)`，codec 解码得到腕局部 `q_h(t)`，再以腕旋转得到 `q_h^B(t)=R_h^B(t)q_h(t)`；关键点为 `p_h^B(t)+q_h^B(t)`。
2. 左栏（GT 视频）用 GT 相对相机 `C_gt(t)=(R_gt(t),p_gt(t))=H_bk⁻¹H_t`：投影 `project(K_out, R_gt(t)^T (p_h^B(t)+q_h^B(t)-p_gt(t)))`。
3. 右栏（生成视频）用预测相机 `C(t)`：把 `p_h^B(t)+q_h^B(t)` 从 B 变换到时刻 t 的预测相机系后投影，用于检查动作、预测相机和生成画面的几何一致性；联合去噪并不保证三者严格一致。
4. 左栏画绿色 GT 与红色预测，二者均用 GT 相机；右栏默认只画红色预测，使用预测相机。内参从 episode 的 `intrinsics.front_1` 出发，按实际缩放、裁剪、补边变换为输出图像的内参；不能直接套用原分辨率内参。
5. 完整投影为 `u=project(K_out, R_C(t)^T (p_h^B(t)+q_h^B(t)-p_C(t)))`。这里 q_h^B 已包含一次腕旋转，不能漏乘或重复乘；先做坐标变换，再透视除法。屏蔽非有限值、相机后方点，屏幕外点不作为有效检测匹配。

- 逐块用完整 `[U_k,V_k]` 经 VAE 解码；按保存的源帧索引取未来帧。连续回放保留上一块预测末帧，不再插入同时间戳的下一块条件帧，不用 GT 覆盖预测末帧；条件图像另作参考展示。初始帧仅显示一次。
- 30fps 回放中，右侧无新 RGB 的时刻标注“背景保持帧”，动作仍按原始时刻更新；这些时刻不用于图像—骨架一致性计分。
- `gt`／`pred_history` 下每块从 GT S_k 重新开始，块边界处预测手会跳回 GT 附近，这是预期现象，不是 bug。`generated` 诊断中 S_(k+1) 取预测末态，轨迹连续。
- 需要全局轨迹图时，`gt`／`pred_history` 用 GT 块首相机位姿左乘；`generated` 只能用预测相机逐块连乘，禁止引入后续 GT 位姿。
- 左栏不重合反映 action 误差；右栏不重合反映“action 与生成视频不一致”，同时包含预测相机误差，预测相机误差另报。逐帧图像—骨架一致性只在真实生成的采样时刻评估；物理指标按原始 30Hz 计算。

## 4. 每卡多样本动态 packing

### 4.1 解除限制的范围

v0.1 限制不只在 DataLoader：官方 replay 校验把一个或两个 video item 当成单样本／control-target，`EgoVerseARModel._build_tf_memory_state` 又要求恰好一个样本；AR attention 也没有跨样本隔离。**不能只增大 `max_samples_per_batch` 或删除断言。**

每个 pack 建立统一元数据：`sample_id、sample-local chunk_id、token_role、文本范围、GEN 范围、U/S/V/A item 映射、clean/noisy KV 范围、条件与有效性 mask、源帧索引、RoPE 起点、两路 timestep`。条件注意力读取 clean pass 的 U_k/S_k；当前 noisy pass 仍保留无 loss 的条件 query 行，预算必须计入这些实际计算行。每个真实 chunk 只有一对 U_k/S_k，不再为首 latent 计 K 个重复 state。

实现要求：

1. 一个 pack 内多个 clip 共享一次 packed clean 前向和一次 packed noisy 前向；不以 Python 逐样本串行前向冒充动态 packing。
2. 修改 replay 布局识别和逐层 K/V 写入／读取，使每个样本按自己的范围访问缓存；多样本不能误识别为单样本双视图或 transfer。每个 clip 仍只有一个视频视图。
3. 文本、视频、action、clean/noisy 流全部按 sample id 隔离，padding 不参与真实 token 注意力或 loss；文本本身的注意力也不能跨样本。
4. RoPE、各块起点状态、chunk 编号、视频与动作加噪、预测解包和 loss mask 均保持样本局部语义。改变样本顺序不改变其内容语义。
5. 所有训练 micro-batch 固定 C=4，历史窗口统一为 H=15 个 chunk，样本的 σ 和噪声独立；支持 T=33/65/129 混合以及最后一个不满 chunk。不引入逐样本不同 C 或历史窗口长度。

**计数口径**：v0.1 的 75K 只计单 pass，不能与本版预算直接比较。本版按两个完整 pass 的实际行数计费（包括 noisy pass 中无 loss 的条件 query），固定 C=4。令 `N=(T-1)/4`、`B=ceil(N/4)`、每个视频 latent 的空间 token 数为 P，则原始预算为 `2×[text_tokens+2+P×(N+B)+8N+B]`；每个 pack 另预留 512 padding，准入必须严格小于 60000，最多 4 个 clip。640×368、T=129、100 个文本 token 时，单样本为 19932，两个样本加预留为 40376。预算版本随 C=4 更新，旧 C=1 元数据拒绝混用。更紧的计数不等于显存验收通过，必须重新 GPU smoke；旧 60K 测试不能证明新组合不会 OOM。

逐档实测每卡 2、3、4 个 clip 的显存与吞吐，再冻结预算；不预先保证能放 4 个 T=129。75K 上限有余量不等于剩余显存足够，也不等于 GPU 利用率已经测明。梯度累积可补有效 batch，但不能替代多样本 packing。

### 4.2 动态 batch 的 loss 与分布式同步

先在每个样本内部算视频、动作 loss，再按全局有效样本数平均；不能按 pack 等权，也不能让长 clip 因 token 多自然获得更大权重。

两种模态分别归约。设全局有有效视频目标的样本数 N_v、有有效动作目标的样本数 N_a（跨所有 rank 求和），梯度通信对 R 个 rank 取平均，则第 r 个 rank 反传 `R/N_v × Σ_i L_video,i + 1.0 × R/N_a × Σ_i L_action,i`，求和只含本 rank 有对应目标的样本。无有效目标的样本不进入该模态分母，并记录两路计数；正常训练 clip 应同时具有两种目标，此时 N_v=N_a。

梯度累积时，N_v、N_a 各自覆盖整个 optimizer update 的所有 micro-batch，不能平均各 micro-batch 的均值。计划好每个 update 的样本数，保持各 rank 的 forward/backward/collective 次数一致；不额外重复除以累积步数。通过与单卡逐样本参考梯度对照确认最终框架缩放。

无累积时有效 batch 为所有 rank 实际 clip 数之和；例如 8 卡各 4 clip 才是 32，动态 packing 时每步会变化。必须记录实际 clip 数、有效 token 数、显存、clip/s、token/s 和 optimizer updates。

## 5. 新表示基础归一化、整体动作损失与可选二次校准

### 5.1 必需：两套 57D Piecewise-Asinh 基础 normalizer

`fixed_camera_wrist_local_delta_latent_v1` 必须重新拟合两套基础 normalizer；这属于表示定义和训练输入合同，不是可选消融：

- `state_normalizer`：输入 57D 块首 state，即相机常量 9D、双腕完整位姿和双手完整新 codec latent。
- `future_normalizer`：输入 57D future，即相机／双腕的 `Δp,ΔR` 和双手 `Δz`。

两套都保留项目现有的**可逆 `PiecewiseAsinhNormalizer` 形式**，使用 train split、正式 C=4 窗口和实际采样权重按维拟合分位数；不是临时 affine/std 标准化。平移、旋转 6D、手形 latent 等字段分别设置有物理／数值依据的单位 floor，避免近常量维被无限放大。相机 state 的前 9 维按协议恒为 0，不从样本方差拟合；其余维度也不得混成一个共同均值或共同尺度。

不读取、拼接或转换旧 27D／18D state 统计，不复用旧 future 统计，也不把旧 hand codec 的内部标准化当作新 57D normalizer。新 codec 先冻结后，才能生成 z／Δz 并拟合这两套统计。训练与推理的归一化／反归一化必须逐维可逆，报告 train 与 held-out 往返误差、分位数覆盖、异常值和单位 floor 命中情况。

两套 normalizer 的文件分别绑定 representation ID、字段切片、C/K/H、数据清单、拟合种子、新左右手 codec hash 和自身 SHA256；checkpoint 保存同一合同。缺失、使用旧 schema、hash 不一致或左右手 codec 身份变化时立即失败。

**暂缓的是额外二次尺度校准，不是上述必要统计。** `action_scale_calibration=false` 只表示不在 `future_normalizer` 输出之后再叠加一层额外 affine/std 或能量平衡；不能据此跳过新 state/future 57D 的 Piecewise-Asinh 拟合。以后若分项日志证明仍有尺度问题，再以独立实验开启二次校准，并单独保存其统计和开关；不能根据文件是否存在自动启用。

### 5.2 一个整体 action loss

对样本 s，M 为第 2.1 节定义的有效未来动作坐标 mask。新 wrist-local 表示通过跟踪审计后，每条真实 future 行的 57D 都为 1，与 FOV 无关；条件 state、末尾 7D 和结构 padding 为 0。因此分母为 `真实 future 行数 × 57`，末尾不满 chunk 只计真实行。跟踪无效的窗口不会进入 loss。令 `v*=ε−z`：

`L_action,s = Σ_(t,d) M_(t,d) × (v_pred − v*)² / Σ_(t,d) M_(t,d)`。

直接聚合全部有效坐标的分子和分母；**不先对八块分别求均值后再求和**。关闭 `subblock_equal_weight` 还不够：旧实现即使按块维数加权，遇到不同可见帧数也不等价于这个整体 masked MSE，需实现真实的统一 reduction。分母为 0 时不计该模态样本，不能产生 NaN。

loss 单测固定 `v_pred` 和 M，只扰动 M=0 位置的有限目标值，断言 loss 与对预测的梯度不变、该位置梯度为 0。新路径另断言不可见 future 行计入分子和分母、有误差时梯度非零；只翻转 FOV 不改变 loss 或梯度；legacy 仍保留原屏蔽。覆盖两个模型 loss 入口、固定 C=4 和末尾 1／2／3 个 latent 组的不满块。这验证的是 reduction：**不能修改 GT 后重新加噪／前向，再要求总 loss 不变**，因为该输入可能通过联合注意力影响其他有效位置。数据损坏与非有限值按第 2.1 节预检处理，不依赖 `NaN×0` 屏蔽。

最终全局目标只有：

`L_total = 1.0 × L_video + 1.0 × L_action`。

视频沿用现有 flow 目标与噪声权重，在样本内按有效视频目标坐标归约；动作保留 uniform 时间权重。两种模态分别按有效样本平均，系数暂不同时调参。不加手腕、旋转、指尖、几何等辅助训练 loss。

分部位的误差和去噪 MSE 仅作为 detach 日志，必要时低频检查共享参数梯度，不额外反传。坐标等权后，手形占 30/57 个坐标（全部有效时约52.6%）；**这是明确的权重选择，不保证各部位梯度相等**。

训练日志同时写入本地 `loss_metrics.jsonl` 和官方在线 W&B：`loss/video_raw`、`loss/action_raw`、`loss/total`，以及 `loss/action_{camera_translation,camera_rotation,right_wrist_translation,right_wrist_rotation,right_hand_latent,left_wrist_translation,left_wrist_rotation,left_hand_latent}_raw`。八项复用整体 action 的有效坐标 mask，先在样本内求该字段 MSE，再按全局有效样本平均；它们是归一化空间诊断量，不是毫米误差，也不作为八个额外训练目标。当前完整57D监督时，八项按维数 `3,6,3,6,15,3,6,15` 加权除以57应还原整体 action loss；不能直接相加。

## 6. 实施顺序与验收

所有代码实现和测试在远端进行；下表是完整交付计划，不代表已经完成。源码路径相对 `cosmos3_joint_video_hand_pose/`，框架补丁另记入 `packages/cosmos3/UPSTREAM.md`，默认不改变原有非 AR 路径。

**当前实施状态（2026-09-29）**：wrist-local＋Δz代码已落地，表示版本为 `fixed_camera_wrist_local_delta_latent_v1`，此前综合CPU回归441项通过；官方启动器复用及路径覆盖已接通。FOV策略已按用户决定落实方案 A：跟踪有效的全部 future action 行参与监督，legacy 不变，本次回归见 [experiment.md](experiment.md)。冻结PCA15已通过验收；两套57D统计已拟合并完成往返验证，129帧8卡显存及官方保存恢复已短测：新配方固定cuDNN benchmark=false、deterministic=true，恢复数值精确一致；分项action日志已在线核实。257/273档8卡容量短测均通过、无OOM；Tdebug2＋Tdebug6官方HSDP（shard8×replicate2、eth0 TCP）的273混合档6步及16rank保存恢复已通过，data trace／sigma／loss最大差0，原阈值不变。两轮online W&B经API核实；性能与通信口径、完整证据见experiment。正式训练仍由用户另行指定。

新表示只按固定 C=4 重新做 token 计数和 GPU 容量 smoke，再决定是否继续使用历史 60,000 token、每卡最多 4 个 clip 和 512 padding 余量。紧凑 noisy 流仍只保留未来 V/A query，clean K/V 保留梯度；完整 GC 规则保持 `every_n=1, warm_up=0, gc_level=2`。旧 C=1 容量上界不再是新配方依据。

按用户后续安排，推理速度优化及第 6.2 节延迟验收暂缓，不作为当前训练启动门槛；保留其测量方案，不宣称通过。第 1.1 节的检测器诊断仍待完成，不能以训练启动代替诊断结果。

**启动顺序**：先用 `ar_v02_prepare_data --audit-only` 生成不依赖 codec 的源窗口清单（`ar_v02_codec_source_windows_v1`，不能作为训练清单）；复用源窗口采集与几何验收工具，由 `prepare_wrist_local_pca.py` 拟合并验收 wrist-local PCA15；再由冻结 codec 生成 train-only z／Δz，拟合两套 57D Piecewise-Asinh normalizer；随后接通数据、训练、缓存、评测与 checkpoint 合同并做固定 C=4 数值／保存恢复 smoke。全部通过后才具备新表示正式训练准入；下一次正式训练仍由用户另行指定。

| 步骤 | 改动入口 | 必须通过的检查 |
|---|---|---|
| 0. 先定位失败来源 | v0.1 已有回放、离线手部检测与诊断报告 | GT 视频上验证检测器；生成视频偏离 GT 与 action 不一致分开；覆盖率、相机投影限制、局部／累积误差分开报告 |
| 1. 固定轴几何与新手形 codec | 新 action 表示、codec、数据准备与物理域测试 | 验证 `Δp` 加法、`ΔR` 左乘、腕局部 q 正确乘一次腕旋转、`Δz` 累加；codec 来源隔离与 held-out 重建达标；换块 z 不变、物理点一致；旧语义／统计／schema 显式拒绝 |
| 2. 两套基础统计与 joint 单样本 | 归一化、dataset、attention、model、inference | state/future 各 57D train-only Piecewise-Asinh 分位数统计、分字段单位 floor 和 hash 合同；C=4/K=8 边界、RoPE、独立 σ、当前双向依赖及无未来泄漏 |
| 3. 持久推理 KV cache | `src/ar_inference.py`、AR 角色 mask／位置 metadata、框架逐层 K/V 接口 | 逐块 U_k/S_k prefill、30 次只读历史及当前条件的联合去噪、一次 V/A clean refresh；保留前 15 个历史 chunk（当前块不计入）；固定 C=4 逐 chunk／逐步对照，覆盖首次淘汰和末尾不满块；阈值断言和有界缓存 |
| 4. 多样本 packing | DataLoader 配置、AR metadata／attention、官方 replay/KV | 混合长度 pack 与逐样本前向／loss／梯度一致；扰动另一条样本的文本和 V/A 不影响本样本；调换样本顺序等价；变长 pack 的分布式与累积梯度缩放正确 |
| 5. 整体 loss 与 checkpoint 合同 | action、normalization、loss、model、contract | 三类 mask、padding 清零、整体 masked MSE；representation、codec、两套 normalizer、数据清单及字段切片缺失／不匹配均失败；额外二次校准保持关闭 |
| 6. 回放与延迟验收（延迟暂缓） | overlay、evaluation、benchmark、新配置 | 投影使用 `p_wrist+R_wrist·q_local`，腕旋转恰好一次；两路固定 30 步；延迟方案保留但暂不作为训练门槛 |
| 7. 历史加噪（暂缓，按需） | 条件流 K/V、条件 timestep、配置 | p=0 等价、U_k/S_k 保持干净、GT 目标不污染、样本与因果隔离；报告实际噪声分布及自由 rollout 收益 |

数值一致性测试先建立 FP32／确定性小规模参考，再根据 bf16 精度对照确定阈值，写成超阈值即失败的断言；不能只打印误差后退出 0。正式训练前完成 CP1/FSDP8 的 50 步 smoke、原始梯度 NaN/Inf 检查和 checkpoint 保存—退出—恢复—下一步验证。

**每块条件的专项验收**：

- 边界索引：固定 C=4、T=33/65/129 和人为构造的末尾不满块下，U_k/S_k 恰为第一条未来 action 的前一帧 b_k；未来 action 不少行、不重行；条件时间不推动 action 行号。
- 逐块编码：`[b_k, b_k+2, …, b_k+8C]` 编码后首个 latent 为 U_k、其余为 V_k；U_k 与该帧单独编码的结果一致。
- 坐标系：S_k 相机槽为 0；相机／双腕用固定 B 轴的 `Δp` 加法和 `ΔR` 左乘，积分和换块后的旋转均通过同一 rotation6D helper 正交化，合成轨迹重建必须等于直接变换到 B 的真值。除短链合成测试外，必须覆盖至少 100 块／3200 帧的 FP32 长链并与 FP64 参考比较，同时逐步检查有限值、正交性和 `det(R)=1`；不得通过放宽阈值掩盖漂移。手形测试覆盖非零腕旋转并断言关键点为 `p_wrist+R_wrist·q_local`；连续换块 z 原值继承，并验证重建关键点与直接刚体变换一致。两套 57D normalizer 的 FP32 往返用 `assert_close(atol=1e-5, rtol=1e-4)`。
- 泄漏：固定 U_k/S_k、历史和当前带噪输入 x_σ（含噪声与两路 σ），只扰动 clean 流中的当前 V/A 和未来 U/S/V/A，断言当前 noisy 输出和 U_k/S_k 的 K/V 不变（第 6.1 节阈值）。不能用扰动后的 GT 重新构造带噪输入。单独扰动 U_(k+1)/S_(k+1)（即当前块末帧／末态），确认不回流。另在非退化小模型上验证改变 U_k 或 S_k 会影响当前 V/A。
- `generated`：U_(k+1) 来自生成视频末帧的重编码；S_(k+1) 由预测末态换到新相机轴，手形 z 原值继承。替换未来 GT 不影响自由 rollout。`gt`：条件来自 GT，预测存入独立缓冲。
- checkpoint 保存／恢复 representation ID、新左右手 codec hash、state/future normalizer hash、类型 embedding 和布局版本；缺少或不匹配时显式失败。旧 checkpoint 走旧解包，不能隐式升级。

### 6.1 KV cache 正确性验收

以 v0.1 `ARSampler.forward()` 的“从序列开头截断到当前 chunk 末尾，再做 teacher-forcing 重算”的执行方式构建参考路径，**两条路径都使用 V0.2 联合 mask、每块 U_k/S_k 的角色 mask／时间及 30 步 schedule**。不能要求 joint 与旧 video-first 模型输出相同。固定同一 checkpoint、文本、同输入测试的各块 U_k/S_k、随机噪声、两路 sigma 和精度；保留独立的 cache／reference 开关，并与真实训练打包路径对照。

参考路径保留完整既往序列，用局部窗口 mask 重算每层历史。**淘汰窗口外的 key，不等于抹去保留 key 在过去形成时吸收的信息**：深层历史 K/V 可能携带当时可见的更早上下文。缓存应保留这些已计算的值，不能仅截取最近 H 块从头重算后称为等价；完整前缀参考仍能复现这种多层因果依赖。

验收同时覆盖两种比较：

- **同输入逐步对照**：先比较各块 U_k/S_k prefill 的逐层 K/V；每个 chunk 的全部 30 个噪声时刻，把相同的 U_k/S_k、当前 V/A 和历史送入 cache 与重算路径，对比视频／action flow 输出、Euler 更新结果，以及 clean refresh 各层写入 K/V；断言 noisy 前向前后历史及 U_k/S_k cache 不变。
- **独立完整 rollout 对照**：相同随机种子下两条路径各自构造下一块 U_k/S_k，比较每个 chunk 的条件、最终 latent、action 和保留的历史 K/V，检查数值误差是否累积。不得每步把缓存路径强制重置为参考输出或共用参考末态后声称通过完整 rollout。

新表示测试矩阵固定 C=4；覆盖 H=15 时窗口首次淘汰前、当步及之后，严格断言当前 k 可见历史为 `max(1,k−15)…k−1`，不将当前块计入 15 个历史 chunk；覆盖窗口未填满、chunk 1 退出窗口，以及未来 latent 余数为 1、2、3 的末尾不满块；覆盖两个 sigma schedule 和 GT、pred_history、generated 历史。小模型单测可另用 H=1、2、3 验证通用边界逻辑，但不改变正式 H=15、C=4 配置。GT-history 模式先产生并独立保存预测，再将 GT 作为后续条件写入 cache，不能把模型预测当作 GT cache 或覆盖预测输出。episode 重置时必须清空缓存。

明确检查 k=16 仍读取 chunk 1、k=17 时 chunk 1 整体（U_1/S_1/V_1/A_1）首次淘汰；任意 k 的当前 U_k/S_k 可读且不占历史额度。检查条件 prefill 无重复写入，30 次 noisy 前向不改变条件 K/V，物理槽位复用不改变其边界时间。

阈值必须落实为测试断言而非日志。初始验收配置如下，均在有效坐标上以 FP32 计算误差，并对全张量断言无 NaN/Inf：

| 数值环境 | `assert_close` 门槛 | 相对 L2 门槛 |
|---|---|---|
| FP32 确定性小模型／参考 attention | `atol=1e-5, rtol=1e-4` | ≤1e-4 |
| bf16 实际模型 | `atol=1e-2, rtol=3e-2` | ≤1e-2 |

相对 L2 使用 `norm(cache−ref) / max(norm(ref),1e-6)`；参考 RMS<1e-6 的近零张量改断言最大绝对误差≤对应 atol，避免分母失真。两种模态、各 chunk／各步和各层 K/V 分别通过，不能用全序列均值掩盖局部越界。上述为待实现验证的工程门槛，不是已有实测结论；超限必须失败并定位 mask／位置／精度原因，不能直接沿用 v0.1 的“误差约 1.9% 即通过”。若精度对照证明需调整，先记录证据、修订并冻结门槛，再重新运行完整矩阵，禁止测试时自动放宽阈值。

### 6.2 延迟、前向次数与显存验收

**暂缓执行**：以下是保留的验收方案，不代表已通过，也不作为当前训练门槛。

目标是固定分辨率、C、15 个历史 chunk、文本长度和 30 步下，**相同大小 chunk 的延迟不随 rollout 长度增长**；不以减少步数、蒸馏或截短可见历史实现，不承诺实时。

1. **逐 chunk 端到端计时**：从调用开始、输入历史状态可用，到物理域 action 可交付且 cache 已写入并为下一块准备完成。包括 U_k 编码、S_k 构造／归一化、条件 prefill、mask／metadata、30 步去噪、V/A clean refresh、淘汰、action 解码与必要的设备同步／传输；不得把条件 prefill 或 refresh 留到计时区间外。`generated` 的生成末帧解码—重编码计入下一块。首块另报告文本 prefill，模型加载和首次 kernel 编译单列。离线视频解码／编码、文件写出另报；若阻塞 action 则计入主延迟。保留双栏回放的 30Hz action、时间戳和两栏同步检查。
2. **真实完整运行**：同一空闲 H800、固定软件／精度／分辨率，预热后至少重复 5 次，记录每块的中位数、P95 和原始耗时。每块输出 noisy／clean／condition-prefill／text-prefill 前向次数、当前 query token 数、可见 key 数、缓存 token 数／字节、峰值 allocated／reserved 显存。文本 prefill 单列后，每块均为 1＋30＋1，共 32 次。U_k/S_k 的空间计入各 chunk 槽位，逐步对照的参考计算不混入计时。
3. **V0.1 历史对照**：640×368、T=129、C=4，原视频 20＋action 20 步的 chunk 去噪耗时约 `9 / 11 / 16 / 20 / 24 / 29 / 33 / 38 s`，合计 **179 s**。原报告是分阶段计时 3 步后外推，未包含完整 mask／回放开销；完整旧评测约 3.5–4 分钟。因此并列保留这些原始口径，不能把 179 s 标成精确 action 端到端耗时或承诺某倍加速。V0.2 必须报告 30 步实测；用同一 V0.2 权重、30 步、joint mask 的 cache 开／关对照，单独量化缓存收益，不为重测降低到 20 步。
4. **跨窗口测试**：按第 3.0 节的覆盖边界，除 T=129 短片对照外，增加至少 48 个完整未来 chunk 的流式 latent rollout，仅验收性能／缓存，不评长时质量。覆盖首次淘汰和至少两次 ring buffer 周转，固定文本长度与每块 token 数。只对当前动作块解码并携带有界积分状态，不每块重解码完整动作前缀；视频解码、指标累计和输出缓冲也不得令 action 路径随总长度增长。
5. **可判定门槛**：排除单列的初始编码／prefill／编译和末尾不满块；比较长 rollout 的最早、最晚各 8 个完整块，要求晚段延迟中位数／早段≤1.10。另在窗口填满后比较最早、最晚各 8 块，要求同样≤1.10，并报告全程延迟曲线与线性趋势，首次淘汰不能出现持续抬升。10% 是本版预设的工程抖动容差，不是已测收益；超限即性能验收失败，需解释和修复后重测。固定容量缓存与工作区不得超出预分配上界，reserved 显存的预热后增长要定位，禁止以累积 CPU/GPU 历史掩盖缓存有界性。

质量评测允许独立保存完整输出供回放，但延迟测试的控制路径必须有界；慢速磁盘或回放队列不得无限堆积。缓存与重算数值对照通过后，仍需验证物理域 action 指标没有回退。

### 6.3 训练安排与结果口径

代码、codec、两套 57D 基础 normalizer 和全链路 smoke 通过后，再按用户指令训练一个完整 V0.2，不先做独立训练消融。完整 V0.2：官方 Nano 初始化，固定 C=4，`fixed_camera_wrist_local_delta_latent_v1`，joint 去噪，整体 MSE，每块 U_k/S_k；额外二次尺度校准和历史加噪关闭。

| 项目 | 安排 |
|---|---|
| 完整 V0.2 训练＋第 6.4 节评测 | 必做，确认实际效果 |
| cache／重算（第 6.1 节）、多样本／逐样本（第 4 节）的数值对照 | 必做，检查实现是否正确；实测覆盖与未完成矩阵见 experiment.md |
| 延迟（第 6.2 节） | 用户已要求暂缓；不额外训练，不作为当前启动门槛 |
| 有／无 state、video-first／joint、尺度校准、packing 训练对照、历史加噪等独立训练对照 | 暂缓，出现具体问题再做 |

- 不做消融，就不单独宣称收益来自 state 或 joint，只评价整套方案。
- v0.1 `sft48464` 初始化不同、评测步数不同（20 步），只能旁列，不作因果归因。
- 所有新评测固定 30 步，不降到 20 步。
- 主指标是 `gt` 主评测下训练集／held-out 的块内误差（第 1.2 节）；同时报告视频质量和动作输出延迟。`pred_history`、`generated`、`oracle` 都是另标的诊断，不当主结果。
- 不同的合理未来可能偏离 GT，所以 GT 轨迹误差与视频—动作一致性分开解释。

### 6.4 评测

所有训练组共用以下评测，清单在看结果前冻结。

**主评测 `gt`（逐块 GT 条件）**：验证集是完整视频，每块的 U_k/S_k 和历史 V/A 都取 GT，模型联合生成当前 V_k/A_k，从 GT S_k 解码。
- 这就是 API 的使用方式：每块给真实条件，看模型能否预测接下来一块。它与训练同分布，不需要“预测后再编码”。
- 报告第 1.2 节的块内误差（块首相机系）、块内方向余弦、与“不动”基线的比较、视频 PSNR，以及右栏“action 与生成视频”的投影一致性。chunk 1 与 chunk 2–8 分开列出。

**辅助诊断**：
- `oracle`：同 `gt`，但当前 V_k 固定为 GT，只去噪 action。用于区分“视频生成不准”和“action 读不对视频”（v0.1 的方向余弦 oracle 0.94 / gt 0.48）。
- `pred_history`：每块的 U_k/S_k 仍取 GT，但历史 V/A 用模型自己的预测。每块给真实图像和 state，不代表历史视频／action 也是真值；若部署 API 保留预测历史，这就是对应的场景。与 `gt` 用同一模型、同一种子配对，差值即历史误差的影响，不需要另训一组。
- `generated`：历史与条件都来自模型自身（第 2.0 节表），报告整段漂移与视频退化，反映部署时无真实观测的最坏情况。不作为主结论。

**统计口径**：固定 held-out 窗口，避免大量重叠窗口伪装独立样本。配对比较复用相同采样种子，按 episode 报告结果及不确定性；旧实验只有 4 个 held-out episode 的局限必须保留；当前多任务验证为 119 个 episode，不能混用样本量。单个训练 seed 的结果只作初步机制结论。

**手部指标口径**：新表示以同一源帧、右／左顺序、米制原始 GT 3D 关键点为主目标，先变换到该块 GT 相机系，再与预测关键点比较；输出 `hand_metric_target='raw_gt_keypoints'`。归档保存原始关键点、相机位姿与显式源帧索引，不能从 GT latent 反推原始标注。主 hand MPJPE 包含 codec 重建误差；`decoded_gt_aux_*` 只作辅助，另报 wrist-local 手形误差与腕位姿误差。新表示缺原始 GT 时主指标及 overlay 明确失败，刚体诊断可单独运行。旧归档保留 decoded-GT 诊断口径，不能标成原始 GT。所有有限关键点当前不按 FOV mask 排除，必须同时报告可见性审计。

## 7. 代码导航

当前生产代码保留在 `cosmos3_joint_video_hand_pose/src/`，以 `ar_v02_` 区分 V0.1。整理不改变导入路径、模型布局、30 步采样及 checkpoint 合同。

### 7.1 按数据流阅读

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

### 7.2 配置、验证和上游补丁

- 当前方案模板：`configs/ar_v0_2_fixed_camera.toml`，显式选择新表示；新 codec 和统计验收完成前不能正式训练。
- `configs/ar_v0_2.toml`：旧动作表示的多任务回归配置，仅用于旧 checkpoint／生命周期验证。
- 重复的 `_c`、`_multitask` TOML 已合并；旧配置注册名仍保留以解析历史快照，语义不变。不可运行的 `_c_no_chunk_state` 和老版 V0.5 配方已删除；AR V0.1 配方保留为历史入口。
- `tests/test_ar_v02_*.py`：按模块组织的回归；`*_gpu.py` 需要 GPU。状态和整体损失另见 `test_ar_chunk_state.py`、`test_whole_action_loss.py`。
- [scripts/README.md](../../scripts/README.md)：Nano 对照、流式 smoke、梯度和恢复验证的入口及使用边界。
- [packages/cosmos3/UPSTREAM.md](../../packages/cosmos3/UPSTREAM.md)：框架补丁清单；不把项目逻辑继续堆入上游目录。
- `outputs/`：结果、日志及诊断快照，不进入 Git；数据和模型权重不随代码整理移动或删除。

训练配方更新（2026-09-29）：固定 C=4、K=8，即每个完整块预测 32 条源帧 action；15 个历史 chunk 不变。新动作表示及新 codec、state/future 57D Piecewise-Asinh 统计只按正式 C=4 准备。experiment.md 中 C=1…4 的数值矩阵属于旧实现历史，不能作为 `fixed_camera_wrist_local_delta_latent_v1` 的通过证据；既有实验和统计快照不覆写。

### 7.3 本轮实施边界（2026-09-29）

- 在 Cosmos3 官方 `_sft_launcher_common.sh` 上增加必要的项目入口扩展，版本化薄启动脚本；Trainer、优化器、调度器、checkpoint、W&B 仍走官方实现。保存／恢复验收必须经过官方 Trainer；数值夹具仅证明相应算子。
- base checkpoint、VAE、text tokenizer 可从配置／环境变量覆盖，默认路径保留；tokenizer 分目录不是错误，但来源与词表必须核验。
- 配方保持 video/action=1:1、Nano action head 初始化、LambdaCosine、基础LR=2e-5、逐步GC、文本dropout=0.1。2026-09-30用户授权正式配方：action2llm、llm2action、action_modality_embed、action_state_embed、vision_condition_embed均5倍，其余1倍；warmup100/cycle1000/max_iter1000/save_iter500，HSDP8×2，273/257/129/65/33混合档。具体数据hash、停机口径和运行状态见experiment当前节；旧配方/运行快照不改写。
- hand MPJPE 主指标对原始 GT 关键点；decoded-GT 仅辅助。统一源帧、左右手、米与毫米、块首／整体坐标转换，并分别报告腕位姿和 wrist-local 手形误差。
- H=15 不改；现有 T=129 只能覆盖最多7块历史。T=257覆盖15块历史，T=273覆盖首次淘汰，均为模型视频帧数；先审计有效窗口和显存，未验收不切换训练档位。
- palm-in-FOV 不是跟踪有效性。用户已选择方案 A：仅新 wrist-local 路径取消 FOV loss 屏蔽，跟踪有效的 future 行全部监督；`invalid_frames` 继续拒绝跟踪无效的整窗。保留不可见连续段／恢复边界审计与 `hand_visibility` 元数据，可另按 FOV 分组评测，不把缺失标签当有效，也不声称已解决模型预测的积分漂移。
