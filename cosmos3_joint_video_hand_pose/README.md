# 视频与双手动作联合训练

当前入口为 [AR V0.2](../docs/ar_v0.2/README.md)，新动作表示配方为 `configs/ar_v0_2_fixed_camera.toml`，直接基于官方 Nano。新 AE 和归一化产物尚未验收完成，不能启动正式训练。

`configs/ar_v0_2.toml` 仅保留旧动作表示的 checkpoint 兼容与生命周期回归。重复的 `_c`、`_multitask` TOML 已合并到该文件；对应配置注册名保留，避免旧运行快照无法解析。旧 overfit 已停用；AR V0.1 入口保留为历史资料。

`scripts/launch_ar_v0_2.sh` 复用官方共享启动器，`src/train.py` 注册项目配置后调用官方 `cosmos_framework.scripts.train`。正式训练循环、优化器、调度器、保存恢复及基础日志使用官方能力。项目新增联合 chunk 布局、数据和动作适配、条件注意力以及必要的合同检查。

清理范围仅为老版 V0.6 之前；AR V0.1／V0.2 的文档、配置与代码保留，误删的 AR V0.1 文件已恢复为历史资料，未重新验收启动。下一次实验须等用户确定配置，当前不启动训练。
