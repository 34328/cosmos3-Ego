# 纯视频 AR 预览

本分支的 AR 继续预训练统一使用这个网页：顶部先选模型版本，再选训练步数，只列出已完成并发布的结果。切换时保留当前的“训练与推理”或“视频与指标”页面。

统一复用现有 sampler、选样清单、renderer和服务器页面。`preview.py` 生成视频，`metrics.py` 在 CPU 上量化已完成的完整视频，`build_page.py` 构建两页，`serve.py` 只服务允许的页面与媒体。不修改训练。

首页 `index.html` 介绍当前 V0.4 的 H/P 并行训练、GT 历史与 Rollout、连续 VAE 编解码；第二页 `results.html` 展示视频与指标。参数与数值来自实际清单和 manifest，不能在模板里填演示分数。

首页 V0.4 数据概况按正式启动绑定的 `outputs/maintenance/full_segment_retention_49920_20261006/summary.json` 及保留 records 核对：32,010 个 train segment，原始时间跨度 89.518898148 小时。77 类采集环境与 705 个具体任务分别来自 episode manifest 的 `environment_id` / `manifest_task`，仅统计保留片段，不按 caption 猜分类。切换到其他训练版本时，数据介绍应核对该版本实际清单，不能沿用 V0.4 数字。

## 完整视频的量化

固定 train/test 各 6 个完整动作段，两种历史共 24 份结果；按原始文本和 GT 选明显操作，不根据生成结果挑样。已有样本直接复用。这个规模用于观察当前 checkpoint，不能当作全数据集任务完成率。

`metrics.py --selection <selection.json> --output <metrics.json> --workers 4` 读取同一时刻的 `gt.mp4` 和 `generated.mp4`，不重新推理、不降回放帧率。PSNR 排除条件首帧，按真实预测帧的 RGB MSE 池化；OpenCV Farneback 相邻帧光流统一在 320×180 上计算，方向余弦先合并 dot/norm，幅值比使用 RMS。这个相邻帧/RMS 口径与早期联合模型的块端点光流/平均幅值不同，不能直接混表。

整段统计包含块间接缝，另保留块内统计、真正的块间接缝和首帧过渡；按完整预测时长分为前、中、后三等份观察趋势。train/test 与两种 history 独立汇总，不平均逐帧、逐段余弦或 PSNR。指标基于已发布 MP4，含压缩和相机运动；它们反映画面与运动接近程度，不单独判定语义任务成功。

`metrics.json` 保存输入 hash、累计量和采样来源，留在服务器私有输出。构建时必须与 selection hash 一致，网页只嵌入显示数值，不公开原始 report 或内部样本身份。

## 先临时查看，再发布

临时推理、GT历史和接缝测试统一放 `outputs/diagnostics/<version>/step<step>/<purpose>/`，默认只生成视频和来源记录，不调用`build_page.py`、不加入网页服务。查看时仅同步选定MP4到本地`tmp/previews/<version>/step<step>/<purpose>/`，直接在对话里展示。

用户确认最终版本后，才把选定结果放入下面的`outputs/visualization/`发布目录，登记到 `catalog.json` 并更新正式网页；复用已完成视频，不重新推理。服务器继续部署在远端。历史测试保留在诊断目录，不在网页里堆叠。

## 完整动作段

新清单使用 `preview_mode="full_segment"`。默认从 train/test 各选几个拿起、移动、放下动作明显的完整 segment；`length_group=short/long` 表示不同的完整短/长动作段，不能从长片截出短片。保留段起止、整段对应文本及真实回放速度，输出去除VAE对齐补帧。

每个样本产物为 `manifest.json`、`preview.mp4`（左GT／右预测）、`generated.mp4`、`gt.mp4`。manifest记录真实frames/fps、checkpoint/seed/去噪步数/CFG/历史刷新噪声和解码模式。页面展示文本、时长和有解释的采样设置，不展示内部样本身份、路径或骨架。

输出结构：

```text
outputs/visualization/<version>/step<step>/<purpose>/
  selection.json
  train_01/{manifest.json,preview.mp4,generated.mp4,gt.mp4}
  test_01/{manifest.json,preview.mp4,generated.mp4,gt.mp4}
  index.html
  results.html
  metrics.json  # 私有计算结果，不由服务公开
```

旧V0.1的窗口裁剪与`short_preview.mp4`格式只兼容已有产物，不是新预览默认。

## 两种历史必须分开说明

| 模式 | Transformer历史KV | VAE解码前缀 | 结果含义 |
|---|---|---|---|
| `history_mode=generated`、`decoder_mode=predicted_prefix` | 已完成的模型预测块 | 拼接的模型预测latent，整段连续解码 | 只给初始图像的自由生成 |
| `history_mode=gt`、`decoder_mode=gt_prefix_montage` | 对应时间的已完成GT块 | 完整GT前缀＋当前预测块，连续解码后裁出当前绝对RGB区间 | GT锚定的单块预测拼图，当前默认GT诊断显示 |
| 旧GT产物、`decoder_mode=predicted_prefix` | 已完成GT块 | 拼接的模型预测latent | 旧诊断显示，Transformer与decoder历史可能不匹配 |

GT模式必须用官方 `get_data_and_condition(..., vision_condition_indexes=None)` 连续编码整段GT。`[[0]]`会触发只编码首帧的优化、未来latent填零，只适用于自由生成首帧条件。两模式都只将首latent标为干净条件；当前块预测结束后，才能把其GT写入后续历史，不能提前给目标块自身的真实图像。

匹配GT解码的实现从latent零连续解码 `GT[:start] + prediction[start:end]`，按原绝对时间裁出目标块RGB并拼接。不能独立重置VAE解码目标块，也不把RGB重新编码成历史。它在已观察样本上减少重影，仍可能有GT重新锚定引起的跳变，不能证明自由生成能力改善。

新preview按实际history自动选择上述解码模式。旧manifest缺失`decoder_mode`按`predicted_prefix`处理，不能把旧MP4重新标记成新结果；显式selection/window/manifest模式不一致必须报错。

历史比较页可在同一`step<step>/`下引用`full_segments/<id>/`和`gt_history/<id>/`，不复制媒体；清单指定`history_comparison=true`，同一id分别列入两种history，`output_dir`为根内相对目录。成对样本必须保持文本、真实长度、源范围及采样设置一致。GT单模式页也应明确其诊断含义。

## 接缝实验

`decoder_diagnostic.py`复用同一预测，比较预测前缀解码、匹配GT前缀解码和GT连续重建。`boundary_preview.py` / `boundary_sampling.py`另测试锁定上一预测末latent的共享边界；这是额外推理条件，不修改默认sampler。在step1500的一条完整短段上未改善，不设为默认；保留小型证据与独立产物，避免重复试验。

原`/boundary_diagnostic/index.html`及GT历史比较已撤出服务。step1500诊断产物归入`outputs/diagnostics/ar_it2v_v0.2/step1500/`，其中原视频及失败回执保留，不改名冒充新结果、不重复计算。旧页面文件仅作为历史存档，不对外服务。

`boundary_preview.py`完成后仅写视频及记录，不自动构建页面；最终获确认时再调用已有`build_diagnostic_page()`，保留四种显示方式的准确标题。

## 构建与服务

在仓库根、与训练相同PYTHONPATH下：

```bash
/home/lzh/miniconda3/envs/cosmos3/bin/python -m visualization.build_page --selection <purpose>/selection.json
# 已有量化结果时增加 --metrics <purpose>/metrics.json
/home/lzh/miniconda3/envs/cosmos3/bin/python -m visualization.build_page --catalog outputs/visualization/catalog.json
/home/lzh/miniconda3/envs/cosmos3/bin/python -m visualization.serve --catalog outputs/visualization/catalog.json --port 18768
```

`catalog.json` 是服务的明确发布清单，路径相对于它所在的目录；每项读取该目录下的 `selection.json`，版本与步数来自真实清单，不能虚填。`metrics` 仅在该 checkpoint 的量化已经完成时登记。示例：

```json
{
  "default": "ar_it2v_v0.4/step1000/validation_two_history",
  "galleries": [
    {"path": "ar_it2v_v0.4/step500/validation_two_history", "metrics": "metrics.json"},
    {"path": "ar_it2v_v0.4/step1000/validation_two_history", "metrics": "metrics.json"}
  ]
}
```

新结果尚未完成时不要登记。更新清单后重建所有已发布页面，再重启已有网页服务使媒体白名单生效；不用新增端口或复制视频。访问 `/` 进入默认 checkpoint，两个下拉框可回到已发布的其他版本与步数。

构建前核对选定样本的manifest和非空MP4；网页使用根内相对媒体路径、支持历史和train/test/长短切换，同时只播放一个视频。checkpoint、seed、每块去噪次数、CFG、回放帧率及历史刷新噪声必须记录，不能把帧率和模型时间步混用。

网页视频默认以 16 fps 播放，按各样本原帧率设置 `playbackRate=16/fps`，可通过已有速度选择恢复原速或其他倍率。视频文件、原始时间轴和量化指标不变；切换历史保留当前速度，视频仍点击时才加载。

启动服务前检查现有进程及端口。服务只监听远端`127.0.0.1`，经已有连接转发；支持Range/HEAD，只开放清单所列的 index.html、results.html 和选定MP4，catalog、selection、metrics、manifest、latent、日志、权重、目录遍历均不公开。临时测试不登记进发布清单，也不为每次测试重复启动服务。

用户明确要求公网分享时，在网页服务所在节点运行 Cloudflare Quick Tunnel 指向 `http://127.0.0.1:18768`，复用同一页面与媒体白名单。实际链接、进程回执和日志存于 ignored `tmp/public_share/`；复用已存活的隧道，不重复启动。公网链接无需本地转发，隧道进程停止或重建后原链接失效。

公网分享默认开放24小时。`tmp/public_share/expiry.json`记录本次到期时间及隧道的hostname/PID birth tick，由同节点临时到期任务只关闭对应隧道；网页服务和训练继续运行，不自动重新公开。后续新增checkpoint沿用当前链接及到期时间，不因更新页面自动续期。
