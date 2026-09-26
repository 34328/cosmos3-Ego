# 上游同步记录

- 上游仓库：NVIDIA/cosmos-framework（本目录 `packages/cosmos3/` 对应上游仓库根目录）
- 当前同步 commit：`cf5d68c00d97ccd2480a2320ed652b92dec63102`（Release 2026-09-23）
- 同步日期：2026-09-26
- 此前基线：`5d6dedc`（2026-08-12）
- 同步方式：整体覆盖为上游内容；本地补丁在后续提交中重新移植，并记录在下方"本地补丁"一节。
- v0.4/v0.6 的 joint video-action attention mask 未随同步保留；复现原实验请 checkout `8525625`。
