# 纯视频 AR 预览

统一复用现有 sampler、选样清单、renderer和服务器页面。`preview.py` 生成视频，`build_page.py` 构建页面，`serve.py` 只服务清单允许的媒体。不训练、不计算质量指标。

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

现有实验页面 `/boundary_diagnostic/index.html` 保留原四种明确标记的显示，不将其他旧GT视频自动改名或重算。

## 构建与服务

在仓库根、与训练相同PYTHONPATH下：

```bash
/home/lzh/miniconda3/envs/cosmos3/bin/python -m cosmos3_ar_it2v.visualization.build_page --selection <purpose>/selection.json
/home/lzh/miniconda3/envs/cosmos3/bin/python -m cosmos3_ar_it2v.visualization.serve --root <purpose> --port 18768
```

构建前核对选定样本的manifest和非空MP4；网页使用根内相对媒体路径、支持历史和train/test/长短切换，同时只播放一个视频。checkpoint、seed、每块去噪次数、CFG、回放帧率及历史刷新噪声必须记录，不能把帧率和模型时间步混用。

启动服务前检查现有进程及端口。服务只监听远端`127.0.0.1`，经已有连接转发；支持Range/HEAD，只开放index和选定MP4，manifest、latent、日志、权重、目录遍历均不公开。独立实验页通过`--extra-page <purpose>`加入相同allowlist。不默认把视频下载到本地，也不为每次预览重复启动服务。
