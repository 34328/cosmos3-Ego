# 纯视频 AR 预览网页

本目录集中管理纯视频预览：`preview.py` 复用现成 sampler 批量推理，并渲染长片与同次 rollout 的短片；`build_page.py` 将完成的产物组成网页；`serve.py` 提供只监听回环地址的轻量服务。不计算指标，不依赖历史联合模型。

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
