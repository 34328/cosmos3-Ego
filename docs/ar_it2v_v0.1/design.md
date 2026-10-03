# AR IT2V V0.1：纯视频因果适配预训练

历史版本设计：本轮实际为最长97帧的随机短窗口训练，已由用户停止，配置保留供复现。本轮后续只授权完整segment/动态packing修改与必要短测；新的正式配方须用户确认后启动。工程目录 `cosmos3_ar_it2v/`；分支 `ar-it2v-pretrain`；独立 worktree `/mnt/lzh/cosmos-ar-it2v`。旧联合模型不改。

## 方法与边界

迁移 [CMD](https://github.com/nv-tlabs/cmd) 第一阶段的核心：把预训练双向视频扩散模型变成逐块因果、可缓存的 AR 视频模型，使用真实视频和并行 diffusion forcing 监督。不是复现 CMD 全套论文数值，不进行 consistency/self-forcing/context-matched distillation，也不承诺 100 小时达到其大规模预训练效果。

- 官方 Cosmos3-Nano DCP 起点；冻结 VAE 和理解专家，训练官方 vision-SFT 的 GEN 参数组 `moe_gen/time_embedder/vae2llm/llm2vae`。不加载 joint checkpoint。
- 输入为文本与首帧图像；网络不创建 action 头/状态/action token，数据仅解码 RGB。
- 连续 VAE 编码整段视频。时间布局为 `I0 | 4 | 4 | …` latent；只首 latent 为干净条件，不再逐块重复编码边界图像。
- 单遍 chunkwise DF；每个未来块独立采样 `u~U(0,1)`，`sigma=clamp(5u/(1+4u),.02,.98)`；块内共享 sigma，首帧 sigma=0。采样覆盖模型方法，底层 RF 配置中的默认分布名称不参与此路径。
- 块内双向、块间因果；总局部窗口16 latent（当前4+历史最多12），无永久首帧 attention sink。packing 的样本互不可见。
- 官方 flow target、timestep 映射与 MSE；首帧从分子/分母排除，所有未来块监督；官方 sample-level reduction 保持每 clip 等权。历史 K/V 与后续 loss 保持可微，无 detach。
- 推理逐块35步，完整块刷新缓存，绝对 RoPE；默认历史 sigma=.02 可配置，清楚记录。它是训练 sigma 支持范围的端点选择，不宣称 CMD 原始 context_noise 的等价值。

## 数据与时空口径

源为 `/mnt/lzh/cosmos-EgoWAM/training_manifests/mecka_100h_v1_{episodes,segments}.csv`，保持原 train/test split，使用 `text_normalized`，不沿用旧 joint 数据筛选。源码 Git 不纳入视频或清单副本。

| 项目 | train | test |
|---|---:|---:|
| episode数 | 4142 | 153 |
| episode时长/h | 96.4097 | 3.5965 |
| 原caption段 | 32355 | 1180 |
| 可用caption段 | 32338 | 1179 |
| 排除过短段 | 17 | 1 |
| 可用段累计时长/h | 95.5079 | 3.56878 |

按用户最新要求，源30fps，stride1，实际模型30fps，连续逐帧输入、不抽帧、不降速。按各段最大合法档17/33/49/65/81/97 RGB帧（5/9/13/17/21/25 latent），每次epoch随机裁一个窗口，不跨caption、不补尾帧。最大约3.2秒；这是一轮短段 AR 适配，不能直接当18秒生成已解决。

保留640×360视野，底部反射pad8至640×368；VAE得到23×40 latent，官方patchify再补齐到24×40后切patch2，解码输出去除确定性空间padding。不改变时间轴、不拉伸图像。

train档位17/33/49/65/81/97数量673/1805/1930/1821/2008/24101；每epoch实际窗口累计25.7814小时，而非每epoch把95小时全看一遍。随机窗口跨epoch覆盖更多位置。

清单 SHA256：
- episodes `98d8abdff990cc3cf0b39a0d1ac0b0338b8d3be9e7b2ea7ed8757acfdacec41d`
- segments `64655579936e40bf2774c0aca50993f1ef6b53c5fc37a44d70cb752950f9be60`

## 首轮训练配方

使用官方 Trainer、AdamW、LambdaCosine、HSDP 与 DCP、W&B callback；项目只扩展模型注意力/噪声、RGB数据接入和只读监控。

- 两节点各8卡，HSDP shard8×replicate2，CP1，bf16/full activation checkpointing。
- 每rank固定4 clips、grad_accum1，即全局64；25 latent档约6k视觉token/clip。45056为模型安全token上限，packer采用固定样本数模式，不同时设token预算。
- seed42，GEN lr2e-5，AdamW betas(.9,.95)/eps1e-6/wd0；无额外模态学习率倍率。wd0是项目覆盖值，官方默认AdamW的weight_decay为0.1，官方确实实现weight decay；不能把项目值描述为官方限制。
- warmup100，cycle3000，f_min .3，max_iter3000，每500保存；clip_norm1.0，EMA关闭。
- 这是以 Cosmos3 优化器配方为基础的迁移，不是逐字照搬 CMD 的 optimizer/训练规模。
- 正式 W&B online；run名 `ar_it2v_v0_1_ego100h_cmd_stage1`，project `rbs_wam_ar_it2v`，entity `alexlzh431564`。临时 smoke 可 disabled，不能用于证明正式上传。
- 纯视频数值停止：非有限loss/原始梯度立即停止；10步均值loss超过首10步3倍、连续10步resident增长超2GiB停止。梯度裁剪是正常优化算法，只记录触发率，不沿用旧 joint 的“9/10 clip即停”经验规则；OOM不自动重试或改预算，不自动恢复。
- 数据恢复将已验证的通用恢复逻辑独立到 `cosmos3_ar_it2v/dataloader.py`，通过 `RecoverablePackingDataLoader` + `IT2VDataLoaderStateCallback`，真实保存随机窗口/packing pending buffer/worker stream状态。

## 验收与查看

CPU：逐块mask、packed隔离、真实noise/loss/timestep、历史梯度、数据分片与窗口恢复。GPU：单节点20步真实训练+正式DCP保存；完整35步RGB推理，确认可加载和输出有限。训练loss下降只能作为数值信号，不替代动作运动正确性。

先用固定train/test短片看清晰度、运动方向、物体转移，再判断是否值得扩大数据/时长或进入后续分布适配；不同时改action方案。输出遵守 `outputs/visualization/ar_it2v_v0.1/step<step>/<用途>/`。

## 与旧项目彻底分开的设置

用户特别指出旧联合方案可能误导：此实验不继承抽帧、action对齐、固定相机筛选、joint归一化器、旧17块长视频要求。源视频原生30fps连续读取；97RGB来自6个C4未来块（24未来latent）加首帧的长度选择，不是旧joint长度。旧stride2开发短测已停止，保留为废弃开发回执，不能充当当前配方验收或预训练。
