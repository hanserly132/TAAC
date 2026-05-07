# Tokenization v3 与轻量 DIN 改动说明

本文简要记录相较当前 v2 强基线新增的模型侧改动。

v2 强基线指：

```bash
--engineered_feature_groups time,pair
--ns_tokenizer_type rankmixer
--dense_projection_mode group_fusion
--no_din
```

当前 v3 默认配置为：

```bash
--engineered_feature_groups time,pair
--ns_tokenizer_type semantic_rankmixer
--dense_projection_mode ue_separated_fusion
--use_din
--din_seq_recent_steps 50
```

## 1. Semantic RankMixer Tokenizer

在 `model.py` 中新增 `SemanticRankMixerNSTokenizer`，用于替代默认的 `rankmixer` tokenizer。

原 `rankmixer` 的做法是：

```text
所有 fid embedding 拼接成长向量 -> 按固定 token 数等长切块 -> 每块投影为 NS token
```

这种方式虽然能自由控制 NS token 数，但可能把不同语义的字段切在同一个 token 里，也可能把同一组语义字段拆开。

v3 的做法是：

```text
按 NS group 得到语义 group embedding -> round-robin 分配到固定数量 NS token -> 每个 token 内 mean + Linear + LayerNorm + SiLU
```

这样保留了固定 token 数的优点，同时让 tokenizer 更尊重字段分组语义。

默认启用：

```bash
--ns_tokenizer_type semantic_rankmixer
--user_ns_tokens 5
--item_ns_tokens 2
```

保留旧 tokenizer 作为回退：

```bash
--ns_tokenizer_type rankmixer
```

## 2. UE 分离 Dense 投影

在 `model.py` 中新增 `UESeparatedDenseFusion`，用于替代默认的 `DenseGroupFusion`。

v2 中 `DenseGroupFusion` 会把所有 dense group 分别投影后门控融合成一个 dense token。v3 进一步把 user dense 中的 UE 类特征单独拿出来：

```text
pretrain_dense: fid 61,87
```

这两个 fid 被视为 user embedding / UE 类 dense 特征，不再和统计 dense、time dense、pair dense 过早混合。

v3 dense 路径为：

```text
pretrain_dense -> user_ue_token
stat_dense + time_engineered + pair_engineered -> aux_dense_token
user_ue_token + aux_dense_token -> gated dense NS token
```

其中 gated dense NS token 进入 HyFormer 主干；`user_ue_token` 和 `aux_dense_token` 会额外进入最终 MLP。

默认启用：

```bash
--dense_projection_mode ue_separated_fusion
```

回退到 v2：

```bash
--dense_projection_mode group_fusion
```

## 3. 轻量 DIN 路径

在 `model.py` 中新增 `LightDINInterest`，用于建模候选 item 与用户最近历史序列之间的 target attention。

DIN query 使用：

```text
item_ns mean pooling
```

DIN key/value 使用四个序列域的 sequence tokens：

```text
seq_a / seq_b / seq_c / seq_d
```

每个序列域单独计算：

```text
score = MLP([query, seq_token, query - seq_token, query * seq_token])
```

然后 mask padding，只对最近 `din_seq_recent_steps` 个 token 做 attention。每个域得到一个 interest vector，四个域再门控融合成一个 `din_interest_token`。

默认启用：

```bash
--use_din
--din_hidden_mult 2
--din_dropout 0.01
--din_seq_recent_steps 50
```

如需关闭：

```bash
--no_din
```

## 4. 最终 MLP 输入变化

v2 的最终分类器主要使用 HyFormer 输出：

```text
HyFormer_output -> MLP -> logits
```

v3 改为拼接四路信息：

```text
concat(
  HyFormer_output,
  din_interest_token,
  user_ue_token,
  aux_dense_token
) -> MLP -> logits
```

MLP 结构为：

```text
Linear(4*d_model, 2*d_model)
LayerNorm(2*d_model)
SiLU
Dropout
Linear(2*d_model, action_num)
```

这样做的目标是让 HyFormer 主干、候选相关兴趣、UE 表示和辅助 dense 统计各自保留独立表达，再在最后一层融合。

## 5. 训练侧改动

涉及文件：

- `model.py`
- `train.py`
- `run.sh`

主要改动：

- 新增 `SemanticRankMixerNSTokenizer`
- 新增 `UESeparatedDenseFusion`
- 新增 `LightDINInterest`
- 新增 `--use_din / --no_din`
- 新增 `--din_hidden_mult`
- 新增 `--din_dropout`
- 新增 `--din_seq_recent_steps`
- `--ns_tokenizer_type` 默认改为 `semantic_rankmixer`
- `--dense_projection_mode` 默认改为 `ue_separated_fusion`
- `run.sh` 默认加入 `--use_din --din_seq_recent_steps 50`
- `train_config.json` 会保存新增参数，供推理恢复

## 6. 推理侧改动

涉及目录：

```text
Model Evaluation/
```

主要改动：

- `Model Evaluation/model.py` 同步训练侧模型结构
- `Model Evaluation/infer.py` fallback 配置同步为 v3 默认值
- 推理阶段从 checkpoint 的 `train_config.json` 恢复：
  - `ns_tokenizer_type`
  - `dense_projection_mode`
  - `use_din`
  - `din_hidden_mult`
  - `din_dropout`
  - `din_seq_recent_steps`

这样可以保证训练和推理构建出的模型结构一致。

## 7. 效率与风险控制

当前 v3 不修改已经验证有效的 `time,pair` 特征构造。

为了控制训练和推理开销：

- HyFormer 主干仍保持不变
- NS token 数默认仍为 `user=5, item=2`
- UE 分离不会额外增加 HyFormer 中的 dense NS token 数
- DIN 只看最近 50 步，不对完整 512/1024 长序列做额外 target attention
- 默认仍关闭 `torch.compile`

需要注意：当前本机 Python 环境没有 `torch`，只能完成 `py_compile` 静态检查，真实 forward smoke test 需要在平台或有 torch 的环境中验证。

## 8. 回退配置

如果 v3 线上效果不如 v2，或遇到模型结构相关报错，可以用以下参数回退到 v2 模型侧配置：

```bash
--ns_tokenizer_type rankmixer \
--dense_projection_mode group_fusion \
--no_din
```

保留默认特征工程：

```bash
--engineered_feature_groups time,pair
```

## 9. 一句话总结

在 v2 的 `time,pair + DenseGroupFusion` 强基线上，v3 默认加入语义化 tokenization、UE dense 分离建模和轻量 DIN 候选-历史兴趣建模，同时保留 v2 配置作为可回退实验。
