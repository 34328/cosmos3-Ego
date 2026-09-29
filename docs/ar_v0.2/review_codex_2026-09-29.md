# V0.2 改动交接 review（2026-09-28 晚至 2026-09-29）

> 2026-09-30 清理说明：按用户要求，已删除训练前短测的 checkpoint、原始日志、trace、临时文件及12个在线W&B测试run和本地副本。下文测试结论为清理前的真实验收记录，所列旧测试输出路径不再代表文件仍保留。PCA及其来源记录、数据清单、当前默认配方的归一化统计、已有正式训练600/1200步checkpoint和回归测试源码保留。后续按用户明确要求，prepared257/273两套测试统计也已删除；需要复用长档时须重新准备并校验对应统计，不可直接沿用已删除的路径。清理清单及关键验收汇总见远端 `outputs/maintenance/cleanup_pretrain_20260930.json`。

供 Claude Code 对照远端工作树复审。项目：`/mnt/lzh/cosmos-EgoWAM`。本文汇总这一轮主要逻辑改动与清理，不替代 [design.md](design.md)；完整运行证据见 [experiment.md](experiment.md)。工作树含累计未提交改动及新增文件，不能只看 `git diff` 而漏掉 untracked 文件，也不能把全部 diff 都归因于最后一次清理。

**最新实施请先看第7.6–7.7及7.9节。** 第1–6节是白天相机轴手形方案及回退的历史记录，第7.1–7.5节是决策过程；当前手形已经按用户确认改为 wrist-local PCA15＋Δz，取消MLP AE重训，FOV监督已按方案 A 修改，不能把上文旧结论当作现状。

## 1. 先区分“失败”的范围

- 旧表示正式训练已完成 1200 步并保存 checkpoint；问题是效果尚未通过、W&B 误设 disabled 导致视频/action 分项历史丢失，不能写成训练进程未完成。原始目录没有完整 W&B DB；事后总 loss 补传已按用户要求删除，历史指标不能补造。
- 后续官方入口短测暴露了恢复 RNG 和日志均值问题，已修复并复验。
- 今天新动作表示的几何／接口已接通，但新 hand AE 的递归重编码质量仍未通过；没有启动新表示的正式 WAM 训练。

## 2. 主要逻辑修改

下列源码名除注明外均在 `cosmos3_joint_video_hand_pose/src/`。

| 模块／文件 | 修改及原因 |
|---|---|
| 官方训练与配置：`train.py`、`config.py` | 正式路径仍转交 Cosmos3 官方 CLI、Trainer、优化器、scheduler、DCP；取消旧 overfit 配方继承和隐含 4×／5× LR 倍率。当前模板 base LR=2e-5、保存间隔600、online，1200步只是模板，不是新训练授权。手工 smoke 循环仅为数值夹具。 |
| 日志：`wandb_metrics.py`、`model.py` | 恢复原生 callbacks／计时清理，移除全局 wandb.log 白名单过滤；项目仅增量记录 video/action/total，并写本地 `loss_metrics.jsonl`。梯度诊断改为只读、覆盖新增 embedding。通过官方 numerator/denominator 接口修复动态 pack 下原生 logger 二次按样本数加权的问题，未改训练梯度。 |
| 恢复：`dataloader_state.py`、`ar_v02_contract.py` | 加强模型／数据状态检查；恢复 worker iterator 时保护 Python、NumPy、Torch CPU RNG，避免 torchdata 抽 base seed 推进已恢复 RNG，造成 sigma 与连续训练不同。保留原生 DCP，不另写保存引擎。 |
| 上游小补丁 | `packages/cosmos3/.../joint_dataloader.py` 仅对无长度 IterableDataset 返回0，恢复有限流长度语义；`callbacks/grad_clip.py` 将累计触发统计明确改名为 since_process_start，不再暗示跨恢复累计。不声称整个上游零修改。 |
| loss／调度 | 当前目标为 `L_video + L_action`，action 系数由0.7改为1；仍是整体 action MSE。保持视频/action 各自 schedule、固定30次联合更新，不做少步或蒸馏。历史训练仍按原0.7记录。 |
| chunk／条件 | 训练和当前方案推理固定 C=4、K=8：边界0的独立 U/S 条件，预测 action 1…32；下块边界32，预测33…64。初始 state只出现一次，不占未来action槽；历史15个chunk，不含当前块。完整块与末块均保留真实时间戳。 |
| 新刚体动作：新增 `action_fixed_camera.py` | 表示名 `fixed_camera_delta_latent_v1`。相机与双腕在固定块首相机轴中，平移 `dp=p_t-p_prev` 直接相加；旋转 `dR=R_t R_prev^T` 左乘恢复。保持57D：相机9＋右腕9＋右手15＋左腕9＋左手15，pad到64，后7维不参与训练。 |
| 新手形：新增 `codec_fixed_camera.py` | `q=keypoints_B-p_wrist_B` 仅去平移，保留块首相机轴朝向；先 `z=E(q)`，再预测 `dz=z_t-z_prev`。state保存完整z，解码为 `p_wrist+D(z)`，禁止再乘腕旋转。新AE不能沿用旧腕局部AE权重。 |
| 跨块与长链 | generated 的末态必须 decode q→换相机轴→重新encode，不能直接旋转15D latent。FP32长链曾在第40块偏离SO(3)，现积分／换块后用rotation6D helper重新正交化，未放宽阈值。 |
| 新统计与数据：新增 `action_fixed_normalization.py`；`ar_dataset.py`、`ar_v02_prepare_data.py` | state与future分别拟合57D、train-only、按通道的可逆 Piecewise-Asinh统计，设置平移／旋转／latent的尺度下限；相机state槽固定0。新有效窗口审计和产物绑定表示、左右AE、清单及统计hash；拒绝旧产物混用。真实新统计尚未生成。 |
| 统一运行时：新增 `action_representation.py`；`ar_v02_{model,inference,streaming,eval,evaluation,overlay}.py` | 用显式表示分派新旧编解码，检查codec／normalizer／模型一致。generated换块保留新state类型；下一图像条件由完整[U,V]解码取末图再编码，不直接取最后latent。generated评测同时变换相机和骨架，避免坐标系错配。 |
| 流式入口与评测 | 修复uint8 RGB未归一化、非零source_start与训练RoPE原点不一致。hand指标明确标记GT来自解码后的GT latent，**不含GT codec相对原始关键点的重建误差**；不能冒充原始关键点MPJPE或AE验收。 |
| 动态packer：`ar_dataset.py`、`ar_v02_dataloader.py` | 原先模型C=4但预算仍按C=1，导致装不满。统一C=4，版本改为 `joint_chunk_cond_v1_two_pass_full_us_c4_v2`；计入两个完整pass和noisy U/S query，保留512 padding、60K与最多4 clip。两个129帧clip、各100文本token预算为40376，而非旧63512。 |
| 打包恢复／审计 | 拒绝旧或缺失预算版本及旧lookahead，包含空buffer旧checkpoint，避免改变batch边界还声称精确续训。修复 `audit_dataloader.py` 默认指向已删V0.5和入口漏传K=8；V0.1兼容推理要求显式给历史TOML。 |

## 3. 清理了什么，保留了什么

**用户纠正（2026-09-29）：清理对象是老版 V0.6 之前，不是 AR V0.1／V0.2。下列早先清理记录中，删除 AR 文件的操作属于误删，现已撤销：恢复 AR V0.1 四份文档、配置、两个启动脚本及 ar_benchmark.py，也恢复 AR V0.2 的 code_map.md 和9月28日 review。恢复的是 Git 保存版本，不能补证删除前未提交的编辑；历史入口未重新验证可运行，不能据此启动实验。**

- 已删除停用的 V0.5 overfit／AR V0.1／no_chunk_state 配方、V0.1启动脚本和旧 benchmark；清理旧 future_experiments、model_learning图文、V0.6规格、training说明、V0.1文档及 `docs/archive/2026-09-26-egowam/` 旧副本。不是删除所有带旧版本名称的公共代码。
- V0.2重复 `_c`／`_multitask` TOML合并。目前配置目录只留：
  - `ar_v0_2_fixed_camera.toml`：新表示模板，注册名 `rbs_wam_ar_v0_2_fixed_camera_delta_latent_v1`。
  - `ar_v0_2.toml`：旧动作表示回归；旧注册别名保留，以解析历史配置，不偷偷切换语义。
- 合并重复说明到V0.2 README／design／experiment，清理旧code_map和review；本文是用户本次要求的新交接review。
- **保留**官方Cosmos3配置、主要工具脚本、被新代码／历史评测使用的数据和动作公共函数、数据清单、旧统计证据、1200步checkpoint与真实运行快照。没有为了补日志重训，也没有覆盖旧实验数据。
- 验证脚本仍默认旧表示时已写明边界；“旧脚本通过”不等于新表示验收通过。

## 4. 验证证据与未完成项

- 官方旧表示训练生命周期：4步保存→恢复到6步、连续6步对照；修复后8-rank数据和sigma一致，bf16 loss在预设阈值内。在线三项loss与本地JSONL核实一致；最终短测原生loss与项目total最大绝对差约5.96e-8。证据见experiment §8.1。
- 旧表示完整Nano缓存矩阵42/42通过；参考是完整历史前缀＋H=15 mask重算，**不是仅重新编码最近15块**。只证明该参考语义下等价，不证明新表示生成质量。
- 双节点仅完成NCCL通信检查，不等于16卡训练／恢复已验收。
- 新表示主联测147项通过；刚体100块／3200帧对FP64参考通过。最近清理回归34项、补充14项；C=4预算联测138项＋审计4项通过。不同批次存在重叠，不能累计成总数。检查含官方TOML加载、完整／末块、padding及旧预算拒绝；diff格式检查通过。
- **新AE仍失败**：两组10000步试验右／左重建mean分别3.54／4.86mm、3.95／5.13mm，但32次rebase为82.47／117.47mm、47.69／69.45mm。identity重复E/D也漂移，指向codec重复投影不稳定；这是静态压力测试，不是实测WAM rollout。gt／pred_history每块用真实state，不可直接外推成同样递归漂移。
- **尚未完成**：新AE定版、真实新state/future统计、新方案GPU端到端、更大C=4 pack的显存及多节点恢复。旧60K smoke不能证明新组合不OOM。未启动新正式训练；速度优化继续暂缓。

## 5. 请 Claude Code 优先核查

1. 固定轴dp、旋转左乘、手形dz及跨块重编码是否前后一致；新旧表示有没有漏掉的默认入口。
2. C=4预算是否等于真实两遍布局；尾块、padding、恢复游标是否严格一致；新装载量需要怎样的GPU短测。
3. AE／normalizer／数据hash是否从数据准备一直约束到checkpoint和评测，能否绕过新产物准入。
4. 官方日志与本地JSONL、恢复RNG、全局样本均值是否仍一致；旧代码清理有没有删掉依赖或遗留失效引用。
5. 区分实现bug、AE质量问题和缺失验收，不把旧表示测试当新方案通过。先review／修复，不启动正式训练。

## 6. Claude Code 复审意见（2026-09-29）

> 执行状态更正：Singer 随后据此实施的启动／路径改造、长 clip 检查及配方变更已按用户要求撤销；本节原始复审意见保留为待决策建议，不是执行授权。恢复核对与166项CPU回归见 experiment 的“2026-09-29 Singer 回退核验”。

只读复审远端 `/mnt/lzh/cosmos-EgoWAM` 工作树（HEAD `e395cd1` 加未提交改动），没有改代码、没有运行测试。下文行号均相对当前远端快照。

### 6.1 启动与配方：没有按官方方式组织

主链确实交给了官方 CLI：`src/train.py` 注册实验后调用 `cosmos_framework.scripts.train`，Trainer、optimizer、DCP 均为原生实现。官方 CLI 没有注册外部实验的钩子，这层薄入口有必要。问题出在入口之外：

1. **没有版本化启动脚本。** 正式 1200 步和各次验收由 `outputs/` 下的临时 Python 脚本（`start_formal.py`、`run_acceptance_stage*.py`）通过 `subprocess.Popen` 启动 torchrun。官方做法是 `examples/launch_sft_*.sh` 调用 `_sft_launcher_common.sh`。这些运行命令引用的 `ar_v0_2_multitask.toml` 已被删除，无法按原命令复现。
2. **保存恢复验收用了自写训练循环。** `smoke_train.py`（624 行）、`validate_ar_v02_training.py`、`check_ar_v02_resume.sh` 自己循环调用 `training_step`。8.1 节已证明，官方 TOML 加 `trainer.max_iter=4 checkpoint.save_iter=2` 就能在官方 `Trainer.train` 上完成同样的验收，这套自写流程应删除。
3. **路径写死在 `config.py`。** VAE、text tokenizer、数据根目录、normalizer 都是硬编码（`config.py:54-57,106,245-260,284-292`）。TOML 里没有 `[checkpoint] load_path` 和 `[model.tokenizer] vae_path`。官方配方用 TOML 加 `${oc.env:…}` 提供。
4. **配方和官方 Nano action 配方不一致，而文档写成了“移除隐含倍率”。** 官方 Nano action 配方共五个：`action_policy_{droid,libero,libero_all,robocasa}_nano.py` 和 `action_fd_droid_posttrain.py`。

| 项 | 官方 Nano action 配方 | 当前 `config.py` |
|---|---|---|
| loss | Nano 基础配置 `action_loss_weight=10`；各 action 配方把 `loss_scale` 改为 10（video 10 : action 10） | `loss_scale=1, action_loss_weight=1`。比例相同但整体小 10 倍；在 `clip_norm=1.0` 下，裁剪的实际效果不同 |
| action 头学习率 | `action2llm/llm2action/action_modality_embed` 各 5× | `lr_multipliers={}` |
| action 头初始化 | `keys_to_skip_loading` 跳过底座 action 头，重新初始化 | 加载 Nano 底座的 action 头 |
| manual_gc | 每 5 步，level 1 | 每 1 步，level 2 |
| scheduler | LambdaLinear | LambdaCosine |

所以“5× 倍率来自旧实验继承”只说对了一半：官方 Nano action 配方本身就带 5×。取消倍率是偏离官方，不是回归官方。上表中哪些要对齐由用户决定；在决定之前，不能把当前配方称为“官方配方”。学习率 2e-4 是官方按全局 batch 8192 设定的，不能照搬。

**建议**：新增版本化脚本 `cosmos3_joint_video_hand_pose/scripts/launch_ar_v0_2.sh`，写法照 `_sft_launcher_common.sh`，只把 `-m` 换成项目入口。路径全部改由 TOML 和环境变量提供。保存恢复等验收一律用官方入口加短步数覆盖参数，删除 `smoke_train.py` 这套夹具，以及 `outputs/` 下的临时 launcher。

### 6.2 动作表示：刚体部分正确，手形坐标需要重新决策

**相机与双腕正确**，与 design 2.0 一致：
- 转到块首相机坐标系：`action_fixed_camera.py:227-230`；
- 帧间增量 `Δp` 和 `ΔR=R_t R_{t-1}ᵀ`：`:234-235`；
- 解码时平移相加、旋转左乘并重新正交化：`:269-273`；
- 换块：`:240-251`。

**手形的实际定义**（`action_fixed_camera.py:231`）：`q_t = kp_t^B − p_wrist,t^B`。这里减去的是**当前帧**手腕的平移，**没有去掉任何手腕旋转**，坐标轴是**块首相机**轴。旧 codec（`action.py:75-81`）是 `R_wrist,tᵀ (kp_t − p_wrist,t)`，即当前帧手腕的完整局部坐标系。两者都不是“块首帧手腕”坐标系。

后果如下：
1. AE 输入虽然只有 20×3 个 xyz、没有显式 RPY，但同一手形在相机轴下转动时，60 个数全部会变。手的朝向因此隐含在 q 里，15D latent 要分出容量来表示约 3 个自由度的朝向。
2. 关键点重建为 `p_wrist + D(z)`（`:280`），腕部 9D 中的旋转不参与关键点重建。腕旋转和 z 中隐含的朝向成了两份可能互相矛盾的信息。
3. 换块时必须 decode→旋转→encode（`:249-251`）。experiment 0.4 中 32 次换块重编码漂移 47–117 mm，重建误差却只有 3.5–5 mm，这正是这条路径造成的。wrist-local 表示的 z 与坐标系无关，换块时 z 直接继承，不需要重编码。

**建议**（待用户决定）：刚体保持块首相机轴加帧间增量；手形改回当前帧 wrist-local，即 `q = R_wrist,tᵀ(kp_t − p_wrist,t)`，重建为 `kp = p + R_wrist · D(z)`，future 仍用 `Δz`。可以先评估能否直接复用 `artifacts/cosmos3_hand_codecs/v2_4/option_b_mlp15`。在用户决定之前，不要继续调新 AE，也不要生成新 normalizer。

### 6.3 其他

- **固定 C=4 与 H=15 冲突。** 最长 clip 129 帧，共 16 个 latent、4 个 chunk，训练中最多只有 3 块历史。推理时 H=15，相当于 480 帧、约 16 秒的历史，满窗口和淘汰旧块都完全不在训练分布里。需要用户决定：缩小 H，还是加长训练 clip。
- 以下几项我认为方向合理，但**没有逐行核对**，不算复审通过：
  - action loss 权重 0.7→1；
  - 流式推理 uint8 RGB 归一化、`source_start` 与 RoPE 原点分离；
  - 恢复时保护 RNG；
  - packer 预算改按 C=4；
  - 恢复官方日志生命周期。
- AR V0.1/V0.2 误删的文件已恢复，清理范围已按用户纠正改为 V0.6 之前的旧内容。

### 6.4 待用户决策

1. 手形坐标：保留块首相机轴，还是改回当前帧 wrist-local。
2. 配方：6.1 表中哪些项与官方 Nano action 配方对齐。
3. H 与 clip 长度。

以上决定之前，不启动任何新表示训练。

## 7. 用户决策与修改要求（2026-09-29）

### 7.1 手形表示：改为当前帧手腕的局部坐标（用户已确认）

**原理**

1. **“相对手腕的 offset”包含两件事**：从哪一点量起（原点），以及沿哪组坐标轴把这个向量写成数字。当前实现 `q = kp^B − p_wrist^B`（`action_fixed_camera.py:231`）的原点是手腕，这一点没错，但坐标轴仍是块首相机的轴。
   - 例：握拳不动，只把手腕转 90°。食指尖的 offset 沿相机轴写，会从 (0,0,0.1) 变成 (0.1,0,0)；沿手腕自身的轴写，始终是 (0,0,0.1)。
2. **当前实现的三个问题都由此引起**：
   - AE 的输入只有 20×3 个 xyz，没有 RPY，但朝向隐含在这些 xyz 里：同一手形，朝向不同，数字就不同，15D latent 只好分出维度表示朝向。
   - 换块会换一组相机轴，q 的数字随之改变，而 AE 是非线性的，不能直接旋转 z，只能 decode→旋转→encode。每做一次都引入 AE 误差，32 次后漂移达到 47–117 mm（experiment 0.4）。
   - 关键点重建为 `p_wrist + D(z)`（`:280`），腕部 9D 中的旋转没有参与，朝向同时存在于 9D 旋转和 z 中，两者可能互相矛盾。
3. **正确的分工**：刚体负责“手在哪、朝哪”，手形只负责“手指怎么弯”。手形与相机、块首无关，因此用手腕自身的轴表达；“块首参考 + 逐帧增量”只作用在刚体上。两部分合起来，最终关键点仍然落在块首相机坐标系下。

**定义**（t 为块内任一帧，B 为块首相机坐标系）

```
刚体（不变）：p_wrist,t^B、R_wrist,t^B 由块首 S_k 加逐帧 Δp、ΔR 积分得到
手形编码：   q_t = (R_wrist,t^B)ᵀ · (kp_t^B − p_wrist,t^B)      # 当前帧手腕局部坐标
            z_t = E(q_t)
state：     S_k 保存完整 z_b
future：    Δz_t = z_t − z_{t−1}
解码：       z_t = z_{t−1} + Δz_t
            kp_t^B = p_wrist,t^B + R_wrist,t^B · D(z_t)
换块：       只对刚体做坐标变换；z 原样沿用，不做 decode→旋转→encode
```

**修改要求**

1. `action_fixed_camera.py`：
   - 编码时对 q 乘 `R_wrist,tᵀ`；
   - 解码时乘回 `R_wrist,t`；
   - `reanchor_state` 删除手形的 decode→旋转→encode，只保留刚体的换轴与重新正交化。
2. `codec_fixed_camera.py`：`INPUT_FRAME` 等声明改为 wrist-local。已训练的“相机轴”AE 作废，不得加载。
3. 优先评估能否直接复用旧的 wrist-local AE（`artifacts/cosmos3_hand_codecs/v2_4/option_b_mlp15`）。它的输入定义与 `action.py:75-81` 相同。
   - 如果复用：需要补上 held-out 重建误差、训练来源和 hash 绑定。
   - 如果不复用：用 train split 按新定义重训。
4. AE 准入：
   - 32 次递归换块测试不再适用，因为换块已不重编码。
   - 改为检查“同一 z 多次 decode 结果一致”，以及 held-out 重建 mean/P95。
   - 门槛沿用左右手各 mean≤5 mm、P95≤15 mm。
5. AE 定版后，再拟合 state/future 两套 57D normalizer。
6. 更新 design 2.0 和 experiment 0.x 中对应的公式与说明；补充刚体长链 + 手形重建的回归测试。

### 7.2 配方（采纳 Codex 意见，保持现状）

当前配方保持不变，包括以下四项：
- loss 系数 1:1；
- 加载底座 action 头，不使用 5× 学习率；
- cosine scheduler；
- 每步 level 2 GC。

我此前的“×10 后裁剪效果与官方相同”不成立：数据、归一化和 loss 归约都与官方不同，所以即使同样 ×10，梯度大小也不会和官方一致。此外，整体 loss 路径要求系数为 1，只改配置里的数值会报错。

训练中需要观察 `grad_clip` 的触发率：如果几乎每步都触发，或者从不触发，再重新评估 loss 尺度。

### 7.3 历史窗口 H 与 clip 长度（采纳 Codex 意见，更正此前计算）

更正：129 是送入模型的视频帧数，stride=2，对应 257 个源帧、8 个未来 chunk，训练中最多能有 7 块历史。此前第 6.3 节写的“4 个 chunk、3 块历史”，以及“满窗口需要 513 帧、超过 60K”都是错的。

| 模型帧 | 源帧 | 未来 chunk | 覆盖范围 | 预算 |
|---:|---:|---:|---|---:|
| 129 | 257 | 8 | 最多 7 块历史 | — |
| 257 | 513 | 16 | 第 16 块读取完整 15 块历史 | 约 40.2K |
| 273 | 545 | 17 | 第 17 块首次淘汰旧历史 | 约 42.6K |

决定如下：
- 保持 H=15；
- 下一步核查 257 帧和 273 帧两档的有效窗口数量，并做 GPU 显存短测；
- 通过后改为长短 clip 混合训练。

以上两档单个 clip 就接近一整卡的 token 预算，混合训练时要同时报告每卡实际样本数的变化。

### 7.4 第二轮复审补充（2026-09-29 18:40）

> 采纳状态见第7.5节；本节为复审原文，建议不等于代码已实施。其中 visibility 的后续用户决定及实施见第7.7节。

1. **7.1 手形修改尚未落地。** `action_fixed_camera.py` 和 `codec_fixed_camera.py` 最后修改于 09:52/09:54，仍是块首相机轴下的手形定义。
2. **复用旧 AE 前必须确认来源。** 旧 `v2_4/option_b_mlp15` 的 checkpoint 只记录了 `architecture` 和 `input_contract`（已是当前帧 wrist-local，定义正确），没有记录训练用的 episode 清单。当前 heldout 有 119 个 episode，无法证明旧 AE 没有用它们训练。
   - 如果查不到训练清单，就按 7.1 的定义，在当前 train split 上重训。每侧只有 13K 参数，代价很小。
   - 重训后按 `codec_fixed_camera.py` 现有的 sidecar 机制，绑定 hash 并写入 heldout 结果。
3. **启动脚本仍未版本化。** `scripts/` 下只有 `launch_ar_v0_1*.sh`，没有 V0.2 的启动脚本。`smoke_train.py` 今天 17:50 还有修改，仍在使用。按 6.1 的要求：
   - 补一个照 `_sft_launcher_common.sh` 写法的 `launch_ar_v0_2.sh`；
   - 保存和恢复验收改用官方入口加短步数覆盖。
4. **`checkpoint.load_path` 不能再用环境变量覆盖。** 当前写死为 `/mnt/lzh/icl/VideoGen/checkpoints/Cosmos3-Nano-official-dcp`（`config.py:212`），之前的 `${oc.env:BASE_CHECKPOINT_PATH,…}` 被去掉了。
   - 恢复环境变量写法，默认值保持不变。
   - VAE 和 text tokenizer 路径也改为从 TOML 或环境变量读取。
   - 顺带确认 text tokenizer 取自 `/mnt/checkpoints/Cosmos3-Nano/` 而不是 DCP 同源目录，是有意为之。
5. **评测指标需要以原始关键点为准。** 当前 hand MPJPE 的 GT 是 GT latent 解码后的关键点（experiment 0.5），不包含 AE 本身的重建误差。手形改为 wrist-local 后，主指标应改为对比原始 GT 关键点，这样 AE 误差和模型误差都计入；解码 GT 这一口径仅作辅助。
6. **训练使用 `cfg_dropout_rate=0.1`，但推理没有 CFG。** 这不影响正确性，但每 10 个样本中有 1 个在不带文本的条件下训练，推理时却从不使用这种条件。请确认是有意保留（为以后启用 guidance），还是应当关闭。
7. **已核查、无问题的项：**
   - visibility mask：抽查 20 个 train episode，共 5.1 万帧，palm 可见率 R/L 分别为 99.95% / 99.98%。增量表示下被屏蔽的帧可以忽略，暂不处理。
   - 数据与 loss：固定窗口按每 32 帧编码一次，未来 action 数量与 `n−1` 严格对应；U/S 的 σ 为 0；loss 只统计未来 V/A，各样本等权。

### 7.5 第二轮复审采纳结论（主线程，只审查）

本次采纳以下实施要求，不代表已经改代码或授权训练；手形方向按用户已确认的 wrist-local＋Δz 执行，配方及 H／clip 取舍不在本次自动修改范围。

| 7.4条目 | 结论与实施边界 |
|---|---|
| 1. 手形未落地 | 属实。现有代码仍是相机轴手形；后续切换须新增明确的表示版本，贯通训练、state、换块、评测及checkpoint，不能只改INPUT_FRAME。 |
| 2. 旧AE来源 | 采纳来源检查。已检查旁边的manifest：有结构、训练超参数和权重hash，但没有episode清单；这是来源未证实，不是已证明数据泄漏。先追溯原训练清单，无法证明与当前heldout隔离则在当前train上重训，绑定权重／数据／统计hash和heldout sidecar。13K参数不等于数据准备和质量验收无需时间。 |
| 3. 版本化启动 | 采纳薄启动入口与官方Trainer生命周期验收。现有train.py已转交官方CLI；补入口不能重写训练循环。官方共享脚本还依赖调用者目录、固定模块等约定，不能只复制文件或直接source便声称已复用成功。先迁移调用方和必要数值检查，再决定退役哪些smoke代码，不能一删了之。 |
| 4. 路径配置 | 采纳可覆盖路径，默认值与当前一致。DCP目录只有权重不必同时含tokenizer，分目录本身不构成错误；仍需核对tokenizer来源、词表、特殊token与底座一致性。不能仅凭目录名确认匹配，不能直接切到不存在的DCP tokenizer路径。 |
| 5. 原始关键点指标 | 采纳作为主指标，decoded-GT保留为辅助。先按同一源帧、左右手顺序、单位和坐标系对齐原始GT，再计算物理误差；generated同时保持整体坐标变换一致。分别报告腕位姿误差和腕局部手形误差，便于区分AE与模型问题。 |
| 6. 文本dropout | 不是确定bug，也不只是为CFG预留；可作为缺文本条件训练／正则化。当前0.1暂不改，不自行增加推理CFG或额外前向。是否关闭应由训练目标和效果决定。 |
| 7. visibility | 不接受“可见率高即可忽略增量缺口”的通过结论。现有标签是palm-in-FOV，不是跟踪有效性；窗口审计检查有限值等，loss另按当前帧可见性屏蔽手腕和手形。少量被屏蔽的增量仍可能影响后续累加，下一帧增量也依赖上一帧标签。应统计最终有效窗口中的连续不可见段／恢复边界，补缺口用例，再决定窗口过滤或监督策略；不要直接把缺失标签当有效，也不要凭此重写mask。 |

本次未独立重跑“20个episode可见率”抽样，也没有将该抽样外推为全库通过。7.4原文和Claude意见保留，避免把建议、已确认方向和已实施代码混写。

本节“先审计、暂不改mask”是当时的边界；后续用户已明确选择方案 A，现状见第7.7节。

### 7.6 用户授权实施（2026-09-29）

本轮按“本地文档→同步远端→代码与测试”执行。采用第7.5节边界；当前版本明确为 `fixed_camera_wrist_local_delta_latent_v1`，旧相机轴版不作同维度隐式兼容。第7.2配方保持原值，未启动 AE 或 WAM 训练。

| 本轮落地 | Claude Code 核查点 |
|---|---|
| `action_fixed_camera.py`、`codec_fixed_camera.py` | q 用当前腕逆旋转；关键点乘一次当前腕旋转；换块仅刚体换轴、z直接复制。新表示／INPUT_FRAME／codec schema 与旧相机轴权重严格区分。 |
| `action_representation.py`、`action_fixed_normalization.py`、`ar_dataset.py`、`ar_v02_model.py`、`ar_v02_contract.py` | 数据、归一化、模型、保存恢复贯通新版本；未知／旧相机轴版本显式失败，旧 legacy AR 路径保持。禁止仅凭57D或同schema猜语义。 |
| `ar_v02_prepare_data.py` 与现有 AE 准备／验收脚本 | 新增不依赖 AE 的源窗口审计，解决准备流程互相依赖；实际采集窗重新校验原始跟踪，绑定数据来源。最终统计目录一起打包已验证 AE／sidecar，检查哈希，避免配置找不到权重。 |
| `ar_v02_eval.py`、`ar_v02_evaluation.py`、`ar_v02_overlay.py` | 数据集保存原始世界系GT关键点；按显式源帧、右左、米制对齐。主MPJPE含AE误差，decoded-GT辅助；generated不读取后续GT state重置；腕旋转不漏乘、不重复。 |
| 项目 `scripts/launch_ar_v0_2.sh`、官方 `_sft_launcher_common.sh`、`config.py`／TOML | 复用官方启动器及CLI，仅加工作目录／项目模块／PYTHONPATH扩展。base、VAE、tokenizer路径可覆盖；loss1:1、action头加载、无LR倍率、cosine、逐步GC、dropout0.1不变。 |
| 仓库 `scripts/verify_ar_v02_native_training.py` | 官方短步保存恢复检查入口；文件存在、脚本测试与真实生命周期验收分开。不再以手写数值循环当正式Trainer证据；旧数值夹具保留明确用途。 |

**当时审计发现（后续方案 A 见7.7）：**当前744 train／119 held-out全部窗口通过既有跟踪有效性检查，但训练集未来行覆盖范围内右／左手连续FOV缺口最长155／104帧，验证集54／87帧。统计按保留窗口覆盖并集计连续段；重叠窗口的训练曝光数另报，不能当独立样本数。增量在不可见行未监督可影响后续累加；回归测试已证明“恢复可见帧的增量正确”不能自动消除这项误差。7.6实施时未改mask；后续用户已决定取消新表示的 FOV 监督屏蔽。

T=257可用train／held-out片段1202／190，T=273为1126／173，均经过真实跟踪窗口检查；不是显存验收，也未启用长clip档位。旧v2_4 AE查不到episode来源，未批准直接复用；需在当前train上重训并验收，之后重新拟合两套统计。

全部本轮证据集中于远端 `outputs/maintenance/wrist_local_20260929/`；`before/`保存修改前快照。首轮失败保留：旧测试codec未声明新INPUT_FRAME；新增collector误用`len(zarr.Array)`，已修为`shape[0]`。最终联测结果与尚缺的GPU验收见 experiment 当前第0节。没有删除AR V0.1／V0.2历史；顺手修正了根README仍误称“V0.2之前都已删除”的过时文字。

最终综合回归 **441 passed，0 failed（130.38s）**；启动／配置／GC最终增量复核 **16 passed（23.21s）**，两组有重叠，不能相加。官方只读preflight按预期退出2，缺真实新AE／sidecar／统计，`lifecycle_verified=false`。Tokenizer结构和特殊token语义已核对兼容，无法从DCP证明不可变上游revision。源码改动清单与修改前后hash保存在同目录`changes.json`和`implementation.patch`，便于只审本轮而非全部累计工作树。

### 7.7 方案 A：跟踪有效的全部 future action 监督（2026-09-29）

用户已明确选择并授权实施：只在 `fixed_camera_wrist_local_delta_latent_v1` 取消 palm-in-FOV loss 屏蔽。原因是 FOV 表示是否在画面中，不是跟踪是否有效；屏蔽仍有有效标签的 Δp／ΔR／Δz，会留下未受监督的增量并影响后续积分。方案 A 补齐这部分监督，不承诺模型积分误差自动消失。

| 修改文件（代码位于远端项目） | 核查点 |
|---|---|
| `cosmos3_joint_video_hand_pose/src/loss.py` | `whole_action_flow_loss(mask_out_of_fov=True)` 默认保留旧行为；显式关闭后 FOV 不影响分子／分母。条件、末尾7D及结构padding仍排除；旧 `visibility_weighted_action_flow_loss` 未改。 |
| `cosmos3_joint_video_hand_pose/src/model.py` | `_compute_whole_losses` 和 `_compute_flow_matching_loss` 两处都按精确表示版本选择策略；legacy／未声明表示仍使用旧屏蔽。保留真实 `hand_visibility`，不以全1替换。 |
| `tests/test_action_fov_policy.py` | 新增19例：不可见行loss／分母／梯度、FOV翻转不变性、两个模型入口及legacy回归、C=4完整／不满块、条件与padding、非有限输入失败。 |
| `tests/test_fixed_camera_data_pipeline.py` | 新增1例：FOV全0但跟踪有效仍保留全部future及原始可见性；不可见源帧关键点损坏后，现有 `invalid_frames` 使整窗拒绝。 |
| 本目录 `design.md`、`experiment.md`、`README.md`、本文 | 明确新旧路径边界，移除“FOV策略待定”；历史复审原文保留并标注后续决定。 |

两组不重叠CPU回归 **58＋104＝162 passed，0 failed**；详细范围与耗时见 experiment。证据目录 `outputs/maintenance/action_fov_scheme_a_20260929/` 保存修改前快照、日志及本轮patch／hash清单。本轮不改跟踪有效性审计、不改评测主口径、不新增分组评测输出、不训练AE或WAM；本地是文档镜像，四份文档与远端同步并校验哈希，实际代码与测试按AGENTS在远端执行。

### 7.8 训练前待办顺序（Claude Code 记录，2026-09-29）

1. 手形 codec：改用 PCA15，在当前 train split 上拟合（已按后续用户指令完成并验收，见7.9）。
2. 拟合 state/future 两套 57D 统计（已完成并校验，见experiment当前第0节）。
3. GPU 显存短测：257 帧和 273 帧两档 clip。
4. 官方入口短测：保存→退出→恢复，并在正式训练前加入按字段的分项 loss 日志。
5. **双节点训练准备**（用户要求放在正式训练之前）：
   - **并行方案**：采用官方 HSDP，节点内 `data_parallel_shard_degree=8`，跨节点 `data_parallel_replicate_degree=2`。通过 `launch_ar_v0_2.sh` 在两端分别设置 `NNODES=2 NODE_RANK MASTER_ADDR` 启动；节点间 SSH 免密未验证，两端经 MCP 分别启动。
   - **IB 网络不可用**：六台节点都有 8 张 NDR 400Gb IB 卡（mlx5_0–7，端口 ACTIVE），但容器内没有 `/dev/infiniband`。2026-09-28 的双节点 NCCL 测试因此退回 eth0 TCP Socket，8 MiB all_reduce 中位耗时约 12 ms。需先确认平台能否向容器暴露 IB 设备；否则跨节点梯度同步会明显拖慢每步耗时，必须实测。
   - **需要实测的项目**：16 卡单步耗时与单节点对比；16-rank 保存恢复，8-rank 的恢复结论不能外推到 16-rank；每步实际消费的 clip 数；在线 W&B 记录。
   - **训练量**：global batch 约翻倍后，按实际消费量重新确定 max_iter 和保存点，不默认沿用 1200 步。

### 7.9 用户改用PCA15：拟合、验收与runtime接入

取消的MLP AE任务只做过检查，未启动训练。本轮按用户新指令改用左右独立PCA15，不训练MLP或WAM。744个train episode的有效窗口覆盖帧去重后拟合，每侧1,827,090条；119个heldout episode验收，每侧287,891条。右手mean/P95=2.951/6.533mm，左手2.792/6.319mm，均通过原5/15mm门槛；解释方差97.1719%／97.6693%。完整分布、在线链接、产物hash见experiment当前第0节。

| 文件 | 本轮改动与复审重点 |
|---|---|
| `scripts/prepare_wrist_local_pca.py`（项目目录下） | 复用已有采集、来源校验和几何验收工具。FP64米制协方差分解，60维只中心化，15维再标准化；分量符号固定。两侧均通过才生成可加载sidecar；权重、manifest、逐episode指标、数据hash与在线W&B齐全。 |
| 项目 `scripts/prepare_fixed_camera_codec.py` | 新增可选“全部有效窗口源帧并集”采集，重叠帧只计一次；保留原MLP抽样入口默认行为。train／heldout严格分开，记录每个源区间及输入hash，跟踪检查不放宽。 |
| `src/codec_fixed_camera.py` | 新通用loader按显式architecture区分PCA／MLP；PCA组件必须15×60且正交，输入std必须全1。严格检查来源、权重hash、sidecar、重建门槛和固定z重复decode。原MLP专用类保留，禁止将PCA静默当MLP。 |
| `src/action_fixed_normalization.py`、`ar_v02_contract.py`、`ar_v02_eval.py` | 三处改用通用loader，统一覆盖统计、保存恢复和评测；仍按左右顺序及hash绑定，不因维数相同忽略codec身份。 |
| `src/config.py`、仓库 `scripts/verify_ar_v02_native_training.py` | codec路径指向用户指定`v3_wrist_local_pca15_train744/`；统计路径保留独立prepared目录。preflight分别检查两处，不再硬性要求codec和统计同目录。 |
| `src/ar_v02_prepare_data.py` | 复用原两套57D Piecewise-Asinh拟合流程；归档codec时根据显式架构保留PCA文件名，不把PCA命名成MLP。 |
| `src/action_fixed_camera.py` | 统计拟合真实数据暴露的数值bug：通用FP32 4×4逆导致齐次底行漂移。改用严格校验后的解析刚体逆；编码、reanchor和腕相机坐标转换同一规则。没有放宽原1e-7底行门槛或改变刚体／手形表示。 |
| `tests/test_wrist_local_pca.py` 与相关数据／配置／合同／几何测试 | 新增PCA公式、子空间往返、组件篡改、sidecar/hash、严格门槛、采集去重及真实位姿底行回归；旧测试替身改为匹配新loader。原MLP兼容测试保留。 |

PCA产物在 `cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v3_wrist_local_pca15_train744/`；v2_4没有改写。拟合原始执行脚本另存证据目录，hash与manifest一致，后续补强manifest生成逻辑不冒充原执行版本。两套57D统计仅在PCA验收通过后拟合；最新统计及回归结果见experiment。证据和失败记录均在`outputs/maintenance/wrist_local_pca15_20260929/`。未启动WAM，也不把CPU／codec验收写成GPU正式训练已就绪。

最终统计已完成：35,272个train窗口，state 200,496行、future 6,415,872行；119个heldout episode各一完整窗口的往返校验通过，原审计窗口与源清单不变，PCA原件／快照／统计／配置哈希绑定一致。PCA验收按全heldout逐关键点汇总判定；右手逐episode有3个mean超5mm、1个P95超15mm，左手无超标，未隐藏这些尾部结果或据此删数据。全部数值与hash见experiment。


### 7.10 训练前第3、4步：显存、分项日志与官方恢复（已完成，待复审）

本次只启动官方入口的短步验证，未启动正式训练。证据集中在 `outputs/validation/ar_v02_pretrain_steps34_20260929T215824/`，`before/`保留修改前版本。请Claude按本节范围复审，不把累计工作区改动都归为本轮。

| 文件 | 修改原因与核查点 |
|---|---|
| `src/loss.py`、`src/model.py`、`src/wandb_metrics.py` | 原有callback已有8项名称，但整体action分支没有产出这些值，故实际日志缺字段。现在复用目标mask与平方误差，detach后按字段和样本归约，接到既有JSONL／W&B回调；目标仍1:1，不新增反传项。legacy默认不采集，保持旧返回结构。额外接通action sigma均值。 |
| `src/pretrain_probe.py`、`src/config.py` | 新增默认关闭的官方callback扩展，仅测量每rank实际clip／token／同步耗时／显存／数据hash和实际双路sigma；要求grad_accum=1，诊断trace只写本地。不是替代Trainer或日志生命周期。 |
| `src/ar_model.py`、`src/ar_v02_model.py` | 官方入口首跑暴露旧父类把teacher_forcing_frames_per_chunk锁成1，V0.2配置4导致构造失败。分版本校验1/4；V0.2的U/S联合布局不得再走官方旧chunkwise `[V0,V1…VC]`截断，否则会丢目标latent。legacy保持原逻辑，新增完整／尾块实际路径回归。 |
| `src/ar_v02_prepare_data.py` | 原fixed准备只审计129/65/33，不能直接启用257/273。让固定表示准备入口显式接受clip档位，复用原审计、编码、train-only统计与hash绑定；另存本次短测包，未覆盖原正式统计。 |
| `src/config.py` | 同数据／sigma恢复后loss仍有差异，旧cuDNN选核设置未通过原阈值。改用benchmark=false、deterministic=true后8rank×2步全部loss精确一致，故仅新fixed-camera配方固定此设置；legacy仍true/false。 |
| 仓库 `scripts/verify_ar_v02_native_training.py`、新增 `scripts/verify_ar_v02_replay.py` | 旧验收入口的disabled改online，修正与官方JobConfig不一致的输出路径。另加跨rank真实trace／sigma精确断言、loss的atol=1e-6/rtol=1e-5断言，失败即退出。官方DCP模型／优化器／调度器／trainer和8rank数据状态均单独核验。 |
| `tests/test_pretrain_field_logging.py`、`test_ar_v02_replay_check.py`、`test_ar_v02_native_launch.py`、`test_fixed_camera_data_pipeline.py` | 覆盖日志不改loss／梯度／RNG、不可见行纳入字段统计、字段到JSONL和W&B、恢复对照的失败断言、C4完整／尾块不截断、长档审计准备、确定性配方不改legacy。 |

失败不掩盖：首轮CPU回归的空诊断dict破坏旧调用者`.detach()`假设，已改为默认不添加该key；首个官方GPU任务因C=4构造校验失败，未进入训练；第一次129恢复超阈值，证据保留，未放宽阈值。启用确定性cuDNN后完整基准6步、从第4步恢复5/6均exit0，8rank数据trace、实际sigma及loss最大差0。此为8卡、129帧、当前软硬件的结论，不外推16卡。W&B全部通过API核对，链接与具体显存／耗时见experiment。

CPU结果：日志/FOV/生命周期/启动40 passed；C4相关33 passed；恢复比较器5 passed；长档准备20 passed；配置最终6 passed（63.30s，MCP等待先超时，落盘测试完整通过）。这些集合重叠，不能相加。257/273统计和显存短测均已完成，详见experiment；双节点及正式训练未启动。


资源修正补充：用户指出单核统计浪费空闲开发机资源。`ar_v02_prepare_data.py`新增显式`--workers`、有序多进程编码和进度日志；固定主进程seed／采样顺序，CPU worker单线程，不改样本数或公式。数据准备回归21 passed（4.05s），串行／并行产物逐字节一致；32worker全库没有另行实测。原单核任务已完成，保留结果避免重复计算。根AGENTS.md记录教训及资源检查／并行／吞吐要求。257/273显存分别使用空闲Tdebug6/Tdebug2的单节点8×H800；不是双节点联合训练。状态与分项结果持续写入experiment。


最终交付：129档6步、257档3步、273档3步均无OOM并正常退出，60K预算／最多4clip未变；长档每rank均实际取到对应长clip。全部当前run的17项loss/sigma经API逐步核对本地一致。官方checkpoint完整性通过；8卡129帧确定性恢复逐rank数据／sigma／loss差异0。请重点复审字段归约是否只用于日志、C4布局为何禁用旧截断、cuDNN设置的范围、有序并行是否保持数据身份。长档与双节点恢复、长期稳定性不在本次通过结论内。


### 7.11 训练前第5步：双节点16卡HSDP短测（已完成，待复审）

本节是7.8第5步的后续验收；7.10“多节点未验收”是前序时点的历史记录。仅运行短测，没有正式训练。本轮未改模型、loss、数据加载、checkpoint或官方启动源码，复用现有实现；文档同步更新README／design／experiment及本文。

- 两节点Tdebug2／Tdebug6、各8张空闲H800，HSDP shard8×replicate2、eth0 TCP；仅NCCL_SOCKET_IFNAME与NCCL_IB_DISABLE两项。trace实际进程组证明分片在节点内、复制跨节点。
- 基准连续6步，第4步保存；独立目录从完整checkpoint4恢复5/6。16rank×2步data trace／实际双路sigma／所有loss最大差0，atol=1e-6、rtol=1e-5不变，比较函数显式ranks=16。源帧／文本／action／state有hash，RGB像素没有单独hash。
- 官方DCP模型／optim／scheduler／trainer与16rank数据状态完整。两轮两节点exit0、无OOM，最终GPU均释放。online W&B两run均finished、17项loss/sigma逐步核对本地一致，链接见experiment。
- 实测global batch为57/51/46/48/40/38，不能固定算64；峰值allocated56.43GiB、reserved67.27GiB。每rank每步数据、同步训练段和官方整步分别报告，不混淆checkpoint／首次启动耗时。
- 与单节点273档第2/3步相比，同步训练段净增加1.97/1.40秒，clip约翻倍；数据组合不同，不将净差全部归因通信。恢复轮用官方profiler记录第6步rank0/8的复制组38次all-reduce，通信核总时7.940/2.922秒，未和非NCCL计算核重叠部分0.842/0.388秒；含对端等待，不能直接当额外整步延迟。基准没有开profiler，性能主表使用基准。
- 首次启动错误把Hydra覆盖写成model.parallelism，两节点配置阶段退出1。改为真实model.config.parallelism并先做CPU配置组合验证，保留失败输出；未调整预算或阈值。

证据目录：`outputs/validation/ar_v02_pretrain_step5_20260929T233700/`，含before文档快照、启动参数、配置组合验证、逐rank原始日志、replay_comparison.json、checkpoint验证、W&B API验证和通信trace汇总。请Claude重点检查：恢复对照是否覆盖全部16rank、数据队列恢复是否精确、W&B字段齐全、通信计时是否正确区分重叠与等待、是否存在越界正式训练。短测通过不代表长期稳定性或训练效果已验收。
