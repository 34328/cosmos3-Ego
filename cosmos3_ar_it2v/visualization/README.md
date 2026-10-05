# 纯视频 AR 预览网页

## 接缝诊断（独立实验）

`boundary_preview.py` 只取一个已选完整短 segment；从同一 checkpoint 生成一次原GT历史预测和一次共享边界预测，不更新权重。`decoder_diagnostic.py` 将同一原预测分别连续解码、在完整GT前缀下逐块解码并裁出绝对RGB范围，另提供GT连续重建。GT前缀拼图是单块诊断，不能称为连续自由生成；所有调用均从latent零开始，不重置后单独解码目标块，也不将RGB重新编码。

`boundary_sampling.py` 保留原C4/local16、seed和四个新latent的官方RF solver。从第二个预测块起，在原时间start−1锁定上一预测末latent，并去掉同位置的历史KV，只刷新原完整块。额外条件是新的推理分布，尚未经过训练；本实验不修改默认sampler。PT存档保存显式continuous normalized格式，不能读取旧联合分块latent。

单卡使用官方`python -m torch.distributed.run --standalone --nproc_per_node=1 -m cosmos3_ar_it2v.visualization.boundary_preview`，参数为`--selection <原完整清单> --id train_02 --output <全新purpose目录> --toml <原配置> --checkpoint <真实checkpoint>`。输出目录是原子claim，失败回执保留，不能重复派发。

完成后通过现有服务的`--extra-page boundary_diagnostic`开放独立子页；仍仅允许该页index与清单选定MP4，latent、manifest、日志不发布。原父页面及其产物保持可用。

本目录集中管理纯视频预览：`preview.py` 复用现成 sampler 批量推理；`build_page.py` 将完成的产物组成网页；`serve.py` 提供只监听回环地址的轻量服务。不计算指标，不依赖历史联合模型。

## 完整动作段（当前默认）

新预览使用 `selection.preview_mode="full_segment"`。每个样本的 `length_group` 为 `short` 或 `long`，代表不同的完整短/长 segment；从原始起点到终点读取连续帧，使用该 segment 的完整对应文本。不得从长片中截出短片，也不随机裁窗口。VAE 对齐补帧在输出时去除，保留所有真实帧及原始回放速度。

每个样本只需要 `preview.mp4`（左 GT、右生成）、`generated.mp4`、`gt.mp4` 和 `manifest.json`。manifest 的 `frames` / `fps` 是实际展示的真实长度/帧率，不需要 `short_start_seconds` 或 `short_duration_seconds`。页面的长短按钮筛选不同完整动作段，不切换同一视频的裁剪版。默认 train/test 各三个明显操作片段；只展示标题、完整文本、实际时长及生成设置，不展示内部样本身份。

例如本轮输出到 `outputs/visualization/ar_it2v_v0.2/step1000/full_segments/`。完成三个非空 MP4 后构建页面，服务只开放这三个 MP4 与 index；URL 包含版本、checkpoint 和模式以防切换预览时命中旧媒体缓存。

## 真实历史诊断与生成历史切换

完整动作段可复用已完成的生成历史产物，仅新增真实历史推理。生成历史只给初始图像，后续读取模型自己的输出；真实历史模式在每块预测完成后，用对应时间段的 GT 更新历史记忆，右栏仍播放模型预测。当前目标块不会得到自己的真实图像，真实历史结果不代表无反馈的整段生成能力。两种模式使用相同样本、checkpoint、seed、去噪次数、CFG 和历史刷新噪声。

真实历史必须调用官方 `get_data_and_condition(..., vision_condition_indexes=None)` 连续编码整段 GT；`[[0]]` 是首帧编码优化，会把后续 latent 填零，只适用于生成历史。编码范围与注意力条件分开：两种模式都只将首 latent 标为干净条件，GT 仅在对应块预测完成后写入历史。

当前GT预览仍将预测latent整段连续解码：各目标块使用GT历史预测，但decoder状态来自拼接的预测前缀，两种历史可能不一致。因此接缝重影不直接作为单块能力证据；严格单块诊断需用同一GT前缀连续解码目标块，不能按块重置VAE。

混合页面以共同父目录（例如 `step1500/`）作为根目录，不使用指向根外的媒体链接，也不复制视频：

```text
step1500/
  selection.json
  index.html
  full_segments/train_01/{manifest.json,preview.mp4,generated.mp4,gt.mp4}
  gt_history/train_01/{manifest.json,preview.mp4,generated.mp4,gt.mp4}
  ...
```

清单沿用完整动作段参数，增加 `history_comparison=true` 和 `default_history_mode="gt"`。每个样本成对列入 `windows`，两项使用相同 `id`，分别指定 `history_mode="generated"`／`"gt"`，`output_dir` 分别为 `full_segments/train_01`／`gt_history/train_01`。新 manifest 保存真实 `history_mode`；旧 manifest 的 `history` 字段仍兼容。两字段同时存在必须一致，未记录历史来源的旧产物只认作 generated。

构建器核对配对样本的文本、真实长度、源帧范围及采样参数。页面默认真实历史，提供两种历史来源切换，布局仍为左 GT／右模型预测；完整长短动作段与 train/test 筛选保持不变。服务器继续仅允许清单列出的 MP4 和 index，不发布其他父目录文件。构建与服务命令的根目录改为上述共同父目录即可。

## 历史裁剪模式（仅兼容已有产物）

未指定 `preview_mode` 的旧预览继续使用以下四视频格式；不作为新预览选样标准。

## 产物约定

统一放在 `outputs/visualization/<version>/step<step>/<purpose>/`。一次预览包含：

```text
preview/
  selection.json
  train_01/manifest.json
  train_01/preview.mp4
  train_01/short_preview.mp4
  train_01/generated.mp4
  train_01/gt.mp4
  ...
  index.html
```

`preview.mp4` 为左真实 RGB、右生成 RGB 的同步对照，无骨架；`short_preview.mp4` 是同一次长 rollout 的动作区间原速裁剪，不单独生成。`generated.mp4` 和 `gt.mp4` 是可单独打开的完整视频。

`selection.json` 的页面契约：

```json
{
  "version": "ar_it2v_v0.1",
  "checkpoint_step": 2000,
  "seed": 42,
  "denoise_steps": 35,
  "guidance": 1.0,
  "context_sigma": 0.02,
  "fps": 30,
  "windows": [
    {
      "id": "train_01",
      "split": "train",
      "title": "拿起并放下杯子",
      "caption": "真实动作段的文本描述",
      "output_dir": "train_01"
    }
  ]
}
```

本次固定 train/test 各三个窗口；`split` 使用 `train` 或 `test`。其他选择证据可以留在清单里，但不会发布到网页。

每个 `manifest.json` 至少提供 `frames`、`fps`、`short_start_seconds`、`short_duration_seconds`，分别表示长片帧数、帧率、短片在长片里的起点及长度。真实采样设置使用和 selection 相同的字段名保存；提供时构建器会逐项核对。sample hash、checkpoint 绝对路径等证据留在 manifest，不在界面展示。

## 构建与交付

在仓库根执行：

```bash
/home/lzh/miniconda3/envs/cosmos3/bin/python -m cosmos3_ar_it2v.visualization.build_page \
  --selection outputs/visualization/ar_it2v_v0.1/step2000/preview/selection.json
```

只有所有选择窗口的 manifest 和四个非空视频都存在时才生成 `index.html`。网页内使用相对媒体路径，包含完整清单数据，不依赖网络字体、第三方 JS 或 JSON 请求。caption 按文本显示，内部身份和路径不进入页面。

部署由主入口检查已有服务后执行：服务仅监听服务器回环地址，经已有远程连接转发交付链接，不发布训练目录和权重。不要为每次预览重复启动服务，也不要默认下载本地 HTML 或整套视频。

页面支持 train/test 筛选与全局长短切换；同时只播放一个视频。长短片展示各自真实时长及裁剪区间；每块去噪次数、回放帧率、CFG 和历史刷新噪声分别解释。

主入口确认端口与已有服务后，可运行：

```bash
/home/lzh/miniconda3/envs/cosmos3/bin/python -m cosmos3_ar_it2v.visualization.serve \
  --root outputs/visualization/ar_it2v_v0.1/step2000/preview --port 18768
```

`serve.py` 使用 aiohttp FileResponse 支持 Range 和 HEAD，仅允许 `index.html` 以及清单内每个样本的四个 MP4；manifest、NPZ、权重、日志和目录遍历一律不开放。服务地址固定为 `127.0.0.1`。
