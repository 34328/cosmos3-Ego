# 清理验证记录

日期：2026-09-26；节点：Tdebug1。所有检查通过 MCP Apply SSH 执行，未启动 GPU 训练或生成。

- 两个项目的 shell 入口通过 `bash -n`，Python 源码通过语法解析。
- 活动 Markdown 的相对文件链接无缺失。
- 活动 Python 源码与 shell 脚本无旧 `/mnt/lzh/cosmos/` 引用。
- B3、v0.6、纯视频配置成功导入，episode/segment 清单和 normalizer 路径存在。
- v0.5/v0.6 回放输入中的图片、动作和参考视频路径检查通过。
- V2 artifact 自带 hash validator 通过，冻结资产文件未改动。
- 262 个 JSON 逐一核对备份 SHA256、当前 SHA256，且变更仅为旧根路径替换。
- `git diff --check` 通过。

## 测试

在现有 cosmos3 Python 环境运行以下测试，共 31 项通过：

```bash
export LD_LIBRARY_PATH=''
export PYTHONPATH="$PWD:$PWD/packages/cosmos3"
export CUDA_VISIBLE_DEVICES=''
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
/home/lzh/miniconda3/envs/cosmos3/bin/python -m pytest -q \
  tests/test_action.py tests/test_codec_normalization.py \
  tests/test_dataset_prompt.py tests/test_loss.py tests/test_temporal.py \
  tests/test_inference_adapter.py cosmos3_egoverse_it2v/tests
```

实际分两批执行，分别为 26 项与 5 项；首次默认插件发现的执行超时，禁用第三方 pytest 插件自动加载后完成。依赖有弃用警告。日志保存在 `outputs/maintenance/2026-09-26-path-relocation/tests.log` 和 `inference-tests.log`。

未重新训练、加载完整历史检查点或人工观看全部回放，因此不将本次清理视为 GPU 端到端验收或新的实验效果结论。
