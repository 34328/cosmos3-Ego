# V0.3 实验记录

2026-10-05：用户批准从官方 Nano 重新训练，单目标块 DF、GT 历史采用 LingBot-VA 公开代码噪声、窗口32 latent，四节点32卡、5000步/save500。

设计与配方见 [design.md](design.md)。CPU：17项新临时用例＋21项旧模型/配置回归，共38项通过；覆盖目标分子分母、历史输入与输出两条梯度、未来隔离、部分末块、RNG恢复和官方逐latent timestep路由。新TOML也通过官方schema组合，确认窗口32、5000步、75008预算与不加载训练状态。开发测试均在 ignored tmp，不入Git。

GPU短测、正式训练启动回执及 W&B API 验证待填入。
