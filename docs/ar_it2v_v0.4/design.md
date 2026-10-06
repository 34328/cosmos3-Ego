# V0.4：GT 历史与并行预测监督

## 已确认的训练语义

纯视频 IT2V，从官方 Nano 重新开始。V0.3 单目标 run 已按用户要求在第194步停止；本版已通过50K单节点8卡3步容量短测，四节点正式训练尚未启动；结果与限制见 experiment.md。

完整 segment 只做一次连续 VAE 编码，文本只处理一次。由同一 GT latent 构造历史 H 与预测 P 两条表示；不是把模型生成的视频回填训练历史。

- H：首 latent 干净；50% 样本全部历史干净，另外50%逐 latent 按 LingBot-VA 公开代码抽 timestep index500..999、shift5，对应 σ约.00498.. .83333。H 不进入直接 loss，保留目标经历史隐藏表示回传的梯度。
- P：复用 Cosmos3 官方 `_get_train_noise_level_vision`、`_add_noise_to_input`、flow interpolation 和速度 loss。使用官方 Nano 视频默认 waver、480分辨率shift5；仅将官方采样结果按 C4 映射为块内共享 timestep。首 latent 干净；不再叠加旧版本手写 uniform/shift/clamp[.02,.98]。
- H_k 读取窗口内 H_≤k；P_k 只读取窗口内 H_<k 和本块 P_k。P 不能读取其他块的 P，也不能读取同块或未来 GT-H。所有样本独立隔离，H/P 使用相同绝对时间 RoPE。
- 每段所有非条件 P 块都进入原生 loss，H 输出不进入 loss；保留真实末帧权重和官方按样本平均。不是每段随机监督一个块，也不是只监督固定末块。

窗口仍为 C4/local32，最多7个完整历史块，无永久首帧 sink。历史版本保持原样。本版采用的仍只是 LingBot 的历史/预测分离及噪声机制，并非完整复现其模型和训练配方。

## 资源与数据

65K双流容量短测OOM后，用户批准改用50K源数据token预算（向下按128对齐为49920）、对应新retention清单、完整segment边界和文本，不能为了容纳双分支静默减半数据长度。Transformer实际token数为 `2 × 源序列token数 − 文本token数`，另行记录，50K源输入展开后接近100K，不能混淆源预算与实际计算长度。

继续使用官方 full activation checkpointing。用户明确拒绝 CPU activation offload 和 CP2，保持 CP1。先做接近满预算的真实 GPU 前反传短测，再决定容量是否足够；不自动截短segment、不自动失败重训。暂存的offload探索仅位于ignored tmp，不作为训练代码。

其余已确认配方：四节点32卡，HSDP8×4/CP1（禁止CP2），5000步/save500，seed42，全模型GEN峰值LR1e-4、官方LambdaCosine warmup100/cycle5000/f_min.3、weight_decay.01，UND冻结。正式启动前必须必要短测、Astra/xhigh审查及W&B online/API核实。

2026-10-06曾讨论、现已暂缓的显存方案（未实现）：将同一segment的预测块分两组，分别前向、反向累积，最后只更新一次参数。整段VAE仍连续编码一次，两组沿用同一套预先采样噪声、原整段有效loss分母及官方样本平均。每次反向后释放激活，第二组重算GT历史并保留历史梯度，不能用第一组预测回填或retain_graph保留完整图。历史上下文要与原双流图一致，不能仅凭直接注意力窗口就裁掉多层历史依赖。显存不保证减半，耗时与容量仍待真实短测。

## 代码来源

LingBot-VA公开代码固定commit `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`：[历史加噪与loss](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/train.py)、[双流注意力与forward_train](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/modules/model.py)、[逐层重计算](https://github.com/Robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/distributed/fsdp.py)。公开训练入口属于其发布的联合模型训练代码，不能声称已核实其全部大规模预训练内部设置。

本项目适配：`cosmos3_ar_it2v/model_v04.py`、`attention_v04.py`、`config_v04.py`、`configs/ego100h_parallel_tf.toml`。官方实现：`packages/cosmos3/cosmos_framework/model/generator/omni_mot_model.py`、`diffusion/rectified_flow.py`，以及 `configs/base/experiment/sft/models/nano_model_config.py`。

用户最新决定：先维持单次前向，将源预算降至49920，重新统计保留/降采样/排除范围并做GPU短测，不启用CP2或CPU offload。两次前向方案暂缓，避免历史重算开销。

## 不改变训练语义的局部优化（2026-10-06）

`network_v04.py` 通过官方 `install_attention_dispatch` 构建扩展点，在FSDP/AC包装前仅特化当前network实例，无新增参数或state-dict键，不修改官方源码。

1. H空间patch及P块内重复的timestep只执行一次官方FP32 embedding，再FP32 gather展开、转BF16；保留原噪声值和RNG。不能先转BF16再聚合重复行梯度。
2. Transformer仍处理完整H/P；最终输出投影与官方unpatchify只处理P。H隐藏表示与历史间接梯度保留，H/P编码metadata与P输出metadata分开。

无V0.4训练标记时回退官方推理路径。数学表达式保持一致，但GEMM维度、浮点归约顺序发生变化，不能宣称完整BF16训练逐位相同。当前compile=false；若以后开启fullgraph编译，动态unique需另行验证。官方MLP与RMSNorm融合未混入本轮。


2026-10-06长segment后续压力测试：20步训练成功、显存稳定；最终官方DCP保存出现NCCL显存分配失败，exit1。50K训练容量与checkpoint生命周期须分别验收，保存问题解决前不得据短测自动启动正式训练。详见experiment.md末节。
