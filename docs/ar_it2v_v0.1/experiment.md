# AR IT2V V0.1 实验记录：最长97帧的随机短窗口训练

## 用户停止与口径更正（2026-10-03）

本轮已经执行的是**最长97帧的随机短窗口训练**：每个caption segment每轮随机取一个17–97帧连续窗口，固定每rank 4样本；没有使用完整segment，也没有启用官方token-budget packing。此前“100小时预训练”的称呼只代表数据清单来源，不能表示每轮完整覆盖100小时；这是实现时自行选择且未充分说明的限制。

用户要求停止后，2026-10-03 09:49:54/55北京时间分别向Tdebug5/6本run的torchrun发送SIGINT。最终真实step2511，最新完整保存checkpoint `iter_000002500`；两端exit回执均为1（人为中断），原始输出/权重全部保留，没有恢复或重训。W&B API核实run状态`killed`、最后step2511/video loss 0.1768373；名称已更正为`ar_it2v_v0_1_random_windows_max97`。API还确认`optim/lr`已有2511个记录点（最后6.963257e-6），并非没有上传LR。

后续改为完整segment与官方token预算packing；本轮只做实现及短测，新的正式配方等待用户确认。

## 当前配方

分支 `ar-it2v-pretrain`，worktree `/mnt/lzh/cosmos-ar-it2v`。官方 Cosmos3-Nano 起点，纯视频+文本+首帧，action 模态关闭。使用原100小时 train split 可用32338片段；原始30fps连续逐帧，stride1；97/81/65/49/33/17档。完整设计与hash见 `design.md`。

正式计划：Tdebug5/6，H800 8×2、HSDP8×2/CP1、内网Socket/eth0；全局64 clips、GEN lr2e-5、warmup100/cycle3000/f_min.3、3000步、每500步保存。正式已启动并经W&B API核实，见下方回执。

## 验证

- 新IT2V CPU覆盖38项。根tests全量收集1093项；首轮963 passed/71 skipped，其余59项为新worktree缺历史fixtures或CPU配置测试调用CUDA验证。修正后仅定向复验123项全过，按JUnit身份合并最终1022 passed/71 GPU skipped/0 failed/error。没有第二次全量重算。回执 `outputs/maintenance/ar_it2v_cpu_regression/`。
- 数据：真实train/test RGB读取，官方packing pending buffer保存/恢复后下一批RGB、source indices、text ids、FPS、SequencePlan逐位一致；无action/state读取。正式恢复使用stateful worker及官方DCP callback。
- 20步真实原始30fps训练，Tdebug6×8 H800，全局32（正式全局64）；退出0。首5步视频loss均值0.3429047，末5步0.2899881；step6–20前后向平均9.6147s，不含checkpoint保存；峰值allocated33.1494GiB/reserved40.4258GiB；梯度有限，未触发数值停止。
- 原生DCP `iter_000000020` 已保存：官方latest marker，model/optim/trainer/scheduler metadata与分片、8 rank dataloader状态齐全。目录 `outputs/validation/ar_it2v_gate/smoke_native30fps/runs/rbs_wam_ar_it2v/ar_it2v_v0_1/ar_it2v_v0_1_smoke_native30fps/`。
- 旧开发smoke01在optimizer更新前因packer同时设sample/token两个预算失败，已改固定4 samples；旧stride2 smoke02按用户新要求停止，不是本配方验收、不续训、不混入正式结果。正式仍从官方Nano起点。

真实step20 DCP已在Tdebug4单H800加载并完成97连续RGB/30fps/25latent/6未来块×35步推理，source indices严格连续，GT与生成视频均97×360×640×3，exit0，无OOM。回执 `outputs/validation/ar_it2v_inference_gate/validation_smoke20_native30fps.json`。正式运行信息见下节。短测loss趋势只证明数值可用，不宣称运动质量已改善。

## 分支隔离

按用户要求，本分支删除旧联合工程 `cosmos3_joint_video_hand_pose/`、旧版本文档、根scripts及旧测试；原始内容保留在 `ar-video-action` 分支和Git历史。根README/AGENTS重新面向纯视频任务编写。通用packing恢复和W&B兼容逻辑已独立到新包，没有旧joint导入。保留上游 `packages/cosmos3`，不裁剪其官方实现。

隔离后的根tests仅包含本项目测试，**39 passed**（含新独立packing恢复）；不存在旧joint目录或代码导入。未因纯目录隔离重复20步GPU短测，模型/数据/优化配方未变。

## 正式训练已启动

- 训练源提交 `aa81810b97fb76c9ac267ae557f44afccfb64e96`；配置SHA256 `2b278f6dde29b3685fed89dda1f5cae8aec710750f9b2c57c4800311e089e1b4`。分支隔离后模型/数据/优化配方不变，39项独立CPU测试通过。
- Tdebug5 rank0 / Tdebug6 rank1，各8张H800，master `10.3.12.50:29883`，NCCL Socket/eth0。两端分别经MCP启动，存在launch claim，禁止重复启动或自动恢复。
- 正式输出：`/mnt/lzh/cosmos-ar-it2v/outputs/pretrain/20261002T174227Z/rbs_wam_ar_it2v/ar_it2v_v0_1/ar_it2v_v0_1_ego100h_cmd_stage1`。两端启动/退出/console回执在 `/mnt/lzh/cosmos-ar-it2v/outputs/maintenance/ar_it2v_v0_1_preflight_20261002T174227Z`。
- W&B：[ar_it2v_v0_1_ego100h_cmd_stage1](https://wandb.ai/alexlzh431564/rbs_wam_ar_it2v/runs/7oimekos)，run `7oimekos`。API于北京时间2026-10-03 01:57核实running及连续step1–5的video loss上传，step5=0.3019655；同时上传gradient、step耗时和显存。
- 实际全局64 clips，step5前后向9.9318s，allocated33.1915GiB/reserved42.0449GiB（各rank最大）；无action参数。首步初始化41.66s，不用于稳态估算。
- 3000步、每500保存，从官方Nano起点重新训练。当前仅启动与数值验收，不宣称画质已改善。完整启动回执见 [evidence/formal_startup.json](evidence/formal_startup.json)。

## iter2000 视频预览

- 使用已完整保存的 `iter_000002000`，train/test 各3个按GT选择的明显操作窗口；原始30fps、seed42、每块35步、CFG1、历史刷新噪声0.02，输入仅首帧与文本。
- 每窗257帧长片（8.57秒）及同次生成的90帧动作裁剪（3秒），GT／生成RGB双栏；长片超出训练最大97帧（3.23秒），不将其当作训练长度内的独立短段评测。本次仅视频观察、不算质量指标。
- Tdebug4 空闲GPU0先验证，GPU1/2/3并行完成余下5窗，6/6退出0、24份MP4；单窗生成及渲染53.39–61.56秒（不含权重加载）。服务器私有网页 `http://127.0.0.1:18768/` 已转发并验证6视频加载，产物位于 `outputs/visualization/ar_it2v_v0.1/step2000/preview/`；完整回执与固定清单见 `evidence/step2000_preview.json`、`evidence/step2000_preview_selection.json`。
