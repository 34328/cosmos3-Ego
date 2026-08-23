# overfit_v0.0 Data Pipeline

`overfit_v0.0` 是最终的 brushing-shoes 小数据过拟合基线。这里两张图只用于说明数据如何流动，不另行定义实现语义。

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
