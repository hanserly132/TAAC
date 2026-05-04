# 特征工程改动说明

本文简要记录相较原 baseline 新增的特征工程改动。

## 1. 新增工程 Dense 特征

在 `dataset.py` 中新增工程特征构造逻辑，并将新增特征追加到 `user_dense_feats` 尾部。

这样做不会改变原有 `user_int_feats / item_int_feats / seq_*` 的输入结构，也不会额外增加 NS token 数，避免破坏当前模型中 `d_model % T` 的约束。

当前默认开启的特征组为：

```bash
--engineered_feature_groups time,pair
```

默认新增约 71 维工程 dense 特征，其中 time 约 59 维，pair 约 12 维。

## 2. 序列时间特征 `time`

针对 `seq_a / seq_b / seq_c / seq_d` 四个行为序列域，新增以下时间统计特征：

- 序列长度
- 最近一次行为距当前样本时间的 recency
- 序列时间跨度
- 行为时间间隔的 mean / min / max
- 最近 `1h / 6h / 1d / 3d / 7d / 30d` 的行为计数
- 样本 `timestamp` 的 hour / day 周期特征
- 周末、深夜 `23/0/1`、周末深夜标记特征

这部分特征不依赖 label，训练和推理都可直接从样本自身计算。当前版本中 recency、窗口计数、span、gap 只使用 `0 < seq_ts <= timestamp` 的历史行为，避免未来时间戳被误当成极近行为。

## 3. Item 侧统计特征 `item`

基于训练集构建 item 统计表，新增：

- `item_id` 曝光次数
- `item_id` 正样本次数
- `item_id` 平滑 CVR
- 每个 `item_int_feats_*` 取值的曝光次数、正样本次数、平滑 CVR

训练阶段只使用训练 RowGroup 构建统计，训练样本 lookup 时做 leave-one-out，降低标签泄漏风险。由于该组属于 target/stat encoding，隐藏测试集分布变化时可能过拟合，当前版本不再默认启用。

对应统计表会保存为：

```text
feature_stats.pkl
```

并随 checkpoint 一起放在 `global_step...` 目录中，供推理阶段加载。

## 4. Pair 特征 `pair`

实现了候选 item 与用户历史序列的匹配特征，当前默认启用，但采用语义白名单以控制噪声和耗时。

默认配置为：

```bash
--engineered_feature_groups time,pair
--pair_recent_steps 20
--pair_seq_fids "seq_a:38,seq_b:69,seq_c:47,seq_d:23"
--pair_candidate_fids "item_id,11"
```

pair 特征包括：

- 候选 item 信息是否命中最近历史行为
- 最近历史中的命中次数
- 最近一次命中的时间差

候选侧使用 `item_id` 和 `item_int_feats_11` 的完整 multi-hot 值；序列侧只匹配 `seq_a:38 / seq_b:69 / seq_c:47 / seq_d:23`，避免跨字段 ID 空间的伪命中。

## 5. Dense 异质分组投影

在 `model.py` 中新增 `DenseGroupFusion`，按 dense 来源分组投影：

- `pretrain_dense`: fid `61,87`
- `stat_dense`: fid `62,63,64,65,66,89,90,91`
- `time_engineered`
- `pair_engineered`
- `item_engineered`，仅显式启用 item 组时存在

每组先独立 `Linear + LayerNorm + SiLU`，再门控融合成 1 个 dense token。这样保持 NS token 数不变，不破坏 `d_model % T` 约束。

## 6. 训练侧改动

涉及文件：

- `dataset.py`
- `train.py`
- `trainer.py`

主要改动：

- 新增 `--use_engineered_features / --no_engineered_features`
- 新增 `--engineered_feature_groups`
- 新增 `--stats_smoothing`
- 新增 `--pair_recent_steps`
- 新增 `--pair_seq_fids`
- 新增 `--pair_candidate_fids`
- 新增 `--dense_projection_mode`
- 仅显式启用 item 组时构建 `feature_stats.pkl`
- checkpoint 保存时同步保存 `feature_stats.pkl`

## 7. 推理侧改动

涉及目录：

```text
Model Evaluation/
```

主要改动：

- `Model Evaluation/dataset.py` 同步训练侧工程特征逻辑
- `Model Evaluation/model.py` 同步 `DenseGroupFusion`
- `Model Evaluation/infer.py` 从 `train_config.json` 恢复 pair 与 dense 投影配置
- 当且仅当训练启用 item 统计特征时，推理阶段加载 `feature_stats.pkl`
- 保证训练和推理的 dense 特征维度与语义一致

## 8. 效率控制

考虑线上真实数据较大，默认只开启：

```text
time,pair
```

默认 pair 使用 4 个高基数 ID-like 序列通道与最近 20 步窗口，避免全量 sideinfo 匹配带来的噪声和额外耗时。`item` 统计特征保留为可选实验，不再默认启用。

## 9. 一句话总结

在原 baseline 基础上，默认改为增强时间特征与语义白名单 pair 特征，并用 DenseGroupFusion 对异质 dense 特征分组投影，同时保留 item 统计特征作为可选实验。
