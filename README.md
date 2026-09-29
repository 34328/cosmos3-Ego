# Cosmos EgoWAM / RBS WAM

当前仅维护 AR V0.2：逐块图像/state 条件、视频/action 联合去噪、30 步采样、15 chunk 历史。

- [设计](docs/ar_v0.2/design.md)
- [实验、验证与已知边界](docs/ar_v0.2/experiment.md)
- [当前状态](docs/ar_v0.2/README.md)
- [验证脚本](scripts/README.md)

正式训练复用 Cosmos3 官方 CLI、Trainer、optimizer/scheduler 和 DCP。项目 `src/train.py` 仅注册配置后转交官方入口；动态 packer 继承官方实现。项目仅扩展 EgoVerse 数据、57D 动作、chunk 条件/注意力/缓存和必要的恢复、指标合同。

清理对象是老版 V0.6 之前的旧系列，不是 AR V0.1／V0.2；AR 两版文档、配置、代码、真实运行快照与 checkpoint 均保留。原始数据及共享预训练权重保留；历史入口未重新验收，不表示允许重新运行。

当前正在修复后的验收阶段；下一轮实验参数待用户指定，不自动训练。远端操作遵循 AGENTS.md。
