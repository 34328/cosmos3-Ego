# AR IT2V V0.1 实验记录

## 当前配方

分支 `ar-it2v-pretrain`，worktree `/mnt/lzh/cosmos-ar-it2v`。官方 Cosmos3-Nano 起点，纯视频+文本+首帧，action 模态关闭。使用原100小时 train split 可用32338片段；原始30fps连续逐帧，stride1；97/81/65/49/33/17档。完整设计与hash见 `design.md`。

正式计划：Tdebug5/6，H800 8×2、HSDP8×2/CP1、内网Socket/eth0；全局64 clips、GEN lr2e-5、warmup100/cycle3000/f_min.3、3000步、每500步保存。正式启动及W&B API核实后补实际回执。

## 验证

- 新IT2V CPU覆盖38项。根tests全量收集1093项；首轮963 passed/71 skipped，其余59项为新worktree缺历史fixtures或CPU配置测试调用CUDA验证。修正后仅定向复验123项全过，按JUnit身份合并最终1022 passed/71 GPU skipped/0 failed/error。没有第二次全量重算。回执 `outputs/maintenance/ar_it2v_cpu_regression/`。
- 数据：真实train/test RGB读取，官方packing pending buffer保存/恢复后下一批RGB、source indices、text ids、FPS、SequencePlan逐位一致；无action/state读取。正式恢复使用stateful worker及官方DCP callback。
- 20步真实原始30fps训练，Tdebug6×8 H800，全局32（正式全局64）；退出0。首5步视频loss均值0.3429047，末5步0.2899881；step6–20前后向平均9.6147s，不含checkpoint保存；峰值allocated33.1494GiB/reserved40.4258GiB；梯度有限，未触发数值停止。
- 原生DCP `iter_000000020` 已保存：官方latest marker，model/optim/trainer/scheduler metadata与分片、8 rank dataloader状态齐全。目录 `outputs/validation/ar_it2v_gate/smoke_native30fps/runs/rbs_wam_ar_it2v/ar_it2v_v0_1/ar_it2v_v0_1_smoke_native30fps/`。
- 旧开发smoke01在optimizer更新前因packer同时设sample/token两个预算失败，已改固定4 samples；旧stride2 smoke02按用户新要求停止，不是本配方验收、不续训、不混入正式结果。正式仍从官方Nano起点。

推理加载门槛与正式运行信息待本轮实际完成后追加。短测loss趋势只证明数值可用，不宣称运动质量已改善。
