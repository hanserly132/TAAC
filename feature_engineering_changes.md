# 特征工程改动说明

本文简要记录相较原 baseline 新增的特征工程改动。

## 1. 新增工程 Dense 特征

在 `dataset.py` 中新增工程特征构造逻辑，并将新增特征追加到 `user_dense_feats` 尾部。

这样做不会改变原有 `user_int_feats / item_int_feats / seq_*` 的输入结构，也不会额外增加 NS token 数，避免破坏当前模型中 `d_model % T` 的约束。

默认开启的特征组为：

```bash
--engineered_feature_groups time,item
```

默认新增约 97 维 dense 特征。

## 2. 序列时间特征 `time`

针对 `seq_a / seq_b / seq_c / seq_d` 四个行为序列域，新增以下时间统计特征：

- 序列长度
- 最近一次行为距当前样本时间的 recency
- 序列时间跨度
- 行为时间间隔的 mean / min / max
- 最近 `1h / 6h / 1d / 3d / 7d / 30d` 的行为计数
- 样本 `timestamp` 的 hour / day 周期特征

这部分特征不依赖 label，训练和推理都可直接从样本自身计算。

## 3. Item 侧统计特征 `item`

基于训练集构建 item 统计表，新增：

- `item_id` 曝光次数
- `item_id` 正样本次数
- `item_id` 平滑 CVR
- 每个 `item_int_feats_*` 取值的曝光次数、正样本次数、平滑 CVR

训练阶段只使用训练 RowGroup 构建统计，训练样本 lookup 时做 leave-one-out，降低标签泄漏风险。

对应统计表会保存为：

```text
feature_stats.pkl
```

并随 checkpoint 一起放在 `global_step...` 目录中，供推理阶段加载。

## 4. Pair 特征 `pair`

实现了候选 item 与用户历史序列的匹配特征，但默认关闭，避免训练和推理过慢。

如需启用：

```bash
--engineered_feature_groups time,item,pair
```

pair 特征包括：

- 候选 item 信息是否命中最近历史行为
- 最近历史中的命中次数
- 最近一次命中的时间差

## 5. 训练侧改动

涉及文件：

- `dataset.py`
- `train.py`
- `trainer.py`

主要改动：

- 新增 `--use_engineered_features / --no_engineered_features`
- 新增 `--engineered_feature_groups`
- 新增 `--stats_smoothing`
- 训练时构建 `feature_stats.pkl`
- checkpoint 保存时同步保存 `feature_stats.pkl`

## 6. 推理侧改动

涉及目录：

```text
Model Evaluation/
```

主要改动：

- `Model Evaluation/dataset.py` 同步训练侧工程特征逻辑
- `Model Evaluation/infer.py` 加载 `feature_stats.pkl`
- 推理阶段使用训练阶段保存的同一份 item 统计表
- 保证训练和推理的 dense 特征维度与语义一致

## 7. 效率控制

考虑线上真实数据较大，默认只开启：

```text
time,item
```

较重的 `pair` 特征默认关闭，避免显著增加每个 epoch 的训练耗时和 30 分钟推理任务压力。

## 8. 一句话总结

在原 baseline 基础上，新增序列时间统计特征和训练集 item 侧平滑统计特征，并保证训练阶段保存统计表、推理阶段复用同一份统计表以保持特征一致。
