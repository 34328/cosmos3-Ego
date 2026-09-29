# AR V0.2

当前方案：`fixed_camera_wrist_local_delta_latent_v1`，57D action，固定 C=4、K=8（每块32条未来action），15个历史chunk，联合去噪30步，`L_video + L_action`。

2026-09-30：双节点1000步正式训练已从`b3585fb`启动，[本次W&B](https://wandb.ai/alexlzh431564/joint_video_hand_pose/runs/bytyfmqa)的首两步11项loss已通过API核实。prepared273在独立正式`t273/`目录重建，11个文件hash与删除前完全一致；273混合档、HSDP8×2、500步保存、指定五模块5倍LR。前100步裁剪只记录，最早110步才按warmup之后的9/10规则停止。实时状态与500/1000步评测口径统一见[experiment当前节](experiment.md#0-当前新表示wrist-local-pca15δz)，下文短测记录属于此前状态。

| 文档 | 用途 |
|---|---|
| [design.md](design.md) | 当前设计；[第7节](design.md#7-代码导航)统一维护代码导航 |
| [experiment.md](experiment.md) | 实验结果、测试证据、失败原因及待完成项 |
| [最新 review](review_codex_2026-09-29.md) | 交给 Claude Code 的改动汇总与核查重点 |

当前手形采用当前帧 wrist-local PCA15＋Δz，换块 z 原值继承；取消MLP AE重训。744个train episode拟合、119个heldout episode验收均完成：右手汇总mean/P95为2.95/6.53mm，左手2.79/6.32mm，两侧通过；右手少数episode尾部误差见experiment。产物在`cosmos3_joint_video_hand_pose/artifacts/cosmos3_hand_codecs/v3_wrist_local_pca15_train744/`，配置与runtime已接通；两套57D统计已拟合并通过119个heldout episode代表窗口的往返校验。FOV监督采用方案 A，legacy 不变。已完成129帧8卡显存短测及官方保存恢复对照：固定cuDNN选核后，恢复数据／sigma／loss精确一致；8项action日志已核实在线上传。257/273帧8卡各3步显存短测也通过、无OOM；每卡实际clip数／峰值显存／耗时和在线链接见experiment。双节点16卡273混合档6步及第4步保存恢复也已通过：16rank数据／sigma／loss差异0，两轮W&B经API核实。通信、每卡clip数、global batch及显存／耗时见experiment，第5步复审重点见review §7.11。当前待Claude复审。仅运行短测，未启动正式训练；下一次正式训练需用户指定。

新方案配置为 `cosmos3_joint_video_hand_pose/configs/ar_v0_2_fixed_camera.toml`；`ar_v0_2.toml` 供旧表示回归。详细参数和指标只在上表文档维护。

历史资料保留在 [旧代码导航](archive/code_map.md)、[9月28日 review](archive/review_codex_2026-09-28.md)，不作为当前配置依据。AR V0.1／V0.2 不属于“老版 V0.6 之前”的删除范围。本地 `data_audit/` 数据审计资料保持原位。

2026-09-30：按用户要求清理了短测输出、临时文件和12个W&B测试run，释放远端约2.02TiB；测试结论保留在文档，原始测试链接已标记删除。PCA／统计／数据清单、已有正式checkpoint和回归测试源码保留，未启动正式训练。
