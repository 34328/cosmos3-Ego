# EgoVerse Joint Video–Action Data Pipeline

这里两张图说明当前 CP1 / FSDP8 / 75K joint baseline 的数据流；57D action、首帧
condition、visibility 与动态 packing 的语义没有因 CP2 guardrail 或 Video-First
mask 改变。图片只作讲解，精确定义仍以模型/数据合同和运行配置为准。

## Full data pipeline

![overfit data pipeline](diagrams/overfit_data_pipeline.png)

矢量版本：[overfit_data_pipeline.svg](diagrams/overfit_data_pipeline.svg)。

## Action data pipeline

![overfit action data pipeline](diagrams/overfit_action_data_pipeline.png)

矢量版本：[overfit_action_data_pipeline.svg](diagrams/overfit_action_data_pipeline.svg)。

读图时只需要抓住三点：

1. `segments.csv` 决定一个 sample 的起止帧；RGB、Pose、visibility 必须共用同一组时间索引。
2. RGB、prompt、action 分三路完成预处理，然后组成一个 WAM sample；Cosmos 原生 packer 只组合已合法的完整 sample。
3. action 先由原始 3D pose 构造为 57D，再补到模型接口的 64D；输出后必须按相反顺序反解回 camera、wrist 和双手 21 点。
