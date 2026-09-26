# Action Contract v2

这是 overfit_v0.0 使用的不可变 action artifact。`v2` 是数据/normalizer schema 版本，不是实验版本。

`normalizers/future_delta_normalizer.json` 内仍保留历史键名 `v6_contract`。该键属于已冻结 JSON 和 SHA256 的一部分，为保证现有 checkpoint、manifest 和 replay 可验证，不能只为重命名而改写；其语义即当前 overfit_v0.0 的 translation-scale 合同。
