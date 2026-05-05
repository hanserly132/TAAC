# Feature Engineering v2 成功经验与原理分析

本文记录当前线上效果较好的版本经验，方便后续继续做特征改动、消融实验和问题排查。

## 版本结论

当前有效版本可以概括为：

```text
baseline_v1 + time,pair 工程特征 + DenseGroupFusion
```

默认开启：

```bash
--engineered_feature_groups time,pair
--pair_recent_steps 20
--pair_seq_fids "seq_a:38,seq_b:69,seq_c:47,seq_d:23"
--pair_candidate_fids "item_id,11"
--dense_projection_mode group_fusion
```

核心经验是一句话：**不要把所有可构造特征都堆进去，而是基于字段语义做少量高置信特征，再用异质 dense 投影让模型更容易吸收这些信号。**

## 重要外部参考：特征 Schema 页面

后续继续改特征时，优先参考这个页面：

```text
https://puiching-memory.github.io/TAAC_2026/analysis/feature-schema/
```

该页面的价值不是替代本项目代码里的 `schema.json`，而是帮助理解字段语义、基数规模和 train/eval/demo/infer 之间的 schema 差异。当前 v2 特征工程里的 pair 白名单、dense 分组和时间字段处理，都参考了这个页面。

### 1. 先区分声明式 schema 与观测 schema

页面强调 schema 有两类：

- 声明式 schema：原始 `schema.json`，描述列布局、FID、词表上界和 multi-hot 维度，是训练/评估/推理加载数据时首先读取的 schema。
- 观测 schema：从真实 parquet 切片统计得到的 sidecar，形状与 `schema.json` 一致，但基数与 multi-hot 维度来自当前数据切片。

这意味着后续做特征选择时要注意：

- 不要只根据 demo_1000 的分布判断真实数据字段强弱。
- 不要把训练日志里的单行 schema payload 当作 train/eval 各自的观测统计。
- 若后续引入依赖基数、multi-hot 长度或字段覆盖率的规则，应优先看 split-specific observed schema sidecar。

### 2. 当前 pair 白名单的 schema 依据

当前 pair 默认只匹配：

```text
seq_a:38
seq_b:69
seq_c:47
seq_d:23
```

这些字段的共同点是：在 schema 参考页中都表现为高基数、ID-like 的序列字段，更适合与候选侧 ID-like 信息做匹配。

候选侧使用：

```text
item_id
item_int_feats_11
```

其中 `item_int_feats_11` 在参考页中是高基数 multi-hot 物品 ID 类特征，因此当前实现会使用它的所有有效 multi-hot 值，而不是只取第一位。

这个依据很关键：pair 特征不是“任意字段数值相等就匹配”，而是“只在语义上可能共享 ID 空间的字段之间匹配”。后续如果要扩展 pair 字段，也应先用 schema 页面确认字段基数和语义，再小步实验。

### 3. 当前 DenseGroupFusion 的 schema 依据

schema 参考页将 `user_dense` 字段列为：

```text
61, 62, 63, 64, 65, 66, 87, 89, 90, 91
```

其中 `61` 和 `87` 被标注为预训练 embedding 向量；其余 dense 字段更像统计、稠密或序列聚合特征。当前分组为：

```text
pretrain_dense: 61,87
stat_dense: 62,63,64,65,66,89,90,91
time_engineered
pair_engineered
item_engineered
```

这个分组不是为了增加模型复杂度，而是为了让不同来源、不同尺度、不同语义的 dense 特征先各自投影，再门控融合。后续新增 dense 特征时，也应先判断它属于哪一类，而不是直接混入一个大 dense 向量里。

### 4. 当前时间特征的 schema 依据

schema 参考页给出了四个序列域的时间戳 FID：

```text
seq_a timestamp fid: 39
seq_b timestamp fid: 67
seq_c timestamp fid: 27
seq_d timestamp fid: 26
```

因此时间特征应按序列域分别计算，而不是把四个域的时间戳混成一条序列。当前实现对每个域分别计算长度、recency、span、gap 和窗口计数，再追加样本级 hour/dow/周末/深夜特征。

页面还提示不同数据集切片中的时间戳统计可能不同，所以后续做时间相关实验时要重点防守异常时间戳：

```text
0 < seq_ts <= timestamp
```

这个过滤条件应保留，除非后续有更明确的时间戳清洗策略。

### 5. 后续使用 schema 页的原则

- 先看字段语义，再看基数，不要只因为基数高就加入特征。
- pair 只在候选侧和序列侧可能共享 ID 空间时使用。
- multi-hot 字段要确认是“多值语义”，不要默认只取第一位。
- dense 新特征要按来源分组，优先复用 `DenseGroupFusion`。
- train/eval/infer 的 schema 可能存在差异，依赖字段覆盖率的实验要小步验证。
- 页面是辅助判断，最终仍以平台真实训练和推理代码可复现为准。

## 为什么这版有效

### 1. 时间特征补足了序列模型不容易直接学稳的全局时间统计

原 baseline 已经有序列编码器，能从序列中学习行为模式，但它未必能高效、稳定地显式表达下面这些统计：

- 当前候选样本距离最近历史行为有多近
- 用户最近 `1h / 6h / 1d / 3d / 7d / 30d` 是否活跃
- 行为序列时间跨度和行为间隔是否集中
- 当前样本发生在一天或一周中的什么位置
- 周末、深夜 `23/0/1`、周末深夜这类周期性场景

这些特征对 PCVR 很自然：转化概率通常与用户近期意图强弱、活跃状态、访问时间段相关。把它们显式追加为 dense 特征，相当于给模型提供稳定的先验统计，减少模型从长序列里自己归纳这些量的负担。

当前实现只使用：

```text
0 < seq_ts <= timestamp
```

的历史行为计算 recency、窗口计数、span 和 gap。这一点很重要，避免未来时间戳或异常时间戳被当作“极近行为”，从而引入噪声甚至隐性泄漏。

### 2. Pair 特征命中了候选 item 与用户历史兴趣的强交叉信号

当前 pair 特征计算的是候选 item 信息是否出现在用户最近历史序列中，包括：

- 是否命中
- 命中次数
- 最近一次命中的时间差

这类信号本质上是候选侧与用户历史侧的交叉特征。推荐系统里，候选 item 或候选 item 的关键 ID-like 属性如果和用户近期历史匹配，往往代表更强的兴趣延续或重复触达关系。

这版有效的关键不是“开启 pair”，而是“只做语义白名单 pair”：

```text
seq_a:38
seq_b:69
seq_c:47
seq_d:23
```

候选侧只使用：

```text
item_id
item_int_feats_11
```

其中 `item_int_feats_11` 是 multi-hot，当前实现会使用所有有效值，而不是只取第一位。

这样做避免了全量 sideinfo 暴力匹配带来的问题：不同字段的 ID 空间可能不一致，即使数值相同也不代表同一语义。全量匹配容易制造大量伪命中，AUC 反而下降。语义白名单让 pair 特征更像“可靠交叉信号”，而不是“随机哈希碰撞信号”。

### 3. DenseGroupFusion 解决了 dense 特征异质性问题

当前 `user_dense_feats` 里混合了多类来源：

- 原始预训练类 dense：`61,87`
- 原始统计类 dense：`62,63,64,65,66,89,90,91`
- 新增 `time_engineered`
- 新增 `pair_engineered`
- 可选 `item_engineered`

这些特征的尺度、来源、语义都不一样。如果简单拼接后用一个 Linear 直接投影，模型需要在同一层里同时处理预训练向量、统计向量、时间统计、pair 命中等异质信号，学习难度更高，也更容易让新增特征和原始 dense 互相干扰。

`DenseGroupFusion` 的处理方式是：

```text
每组 dense -> Linear -> LayerNorm -> SiLU -> gate 融合 -> 1 个 dense token
```

这样有几个好处：

- 每个语义组先在自己的子空间内投影，降低互相污染
- `LayerNorm` 缓解不同组尺度差异
- gate 让模型动态决定每组信号的使用强度
- 最终仍然融合成 1 个 dense token，不增加 NS token 数
- 不破坏 `d_model % T` 等 baseline 主干约束

这也是这版比“单纯追加 dense 特征”更稳的重要原因。

## 为什么默认不再启用 item target/stat 特征

之前的 `item` 统计特征包括曝光次数、正样本次数和平滑 CVR，本质是 target/stat encoding。它在本地验证集上可能有效，但线上隐藏测试集上存在更高风险：

- 统计来自训练集，隐藏测试集分布可能漂移
- 高频 item 的历史 CVR 容易过拟合训练窗口
- 训练和推理必须依赖同一份 `feature_stats.pkl`，维护成本更高
- 即使做 leave-one-out，也不能完全消除 target encoding 的泛化风险

因此当前默认改为 `time,pair`，把 `item` 保留为可选实验。这个选择的经验是：**优先使用不依赖 label、训练推理天然一致的行为时间特征和语义 pair 特征。**

## 维度与效率控制

当前默认新增工程 dense 维度为：

```text
time: 59
pair: 12
total: 71
```

其中 pair 维度来自 4 个白名单序列字段，每个字段输出 3 维：

```text
hit / hit_count_log / recency_log
```

效率控制原则：

- 不全量遍历所有 sideinfo 字段做 pair
- pair 只看最近 `20` 步
- 不新增额外 NS token
- 不默认构建 item target stats
- 默认关闭 `torch.compile`，避免线上 inductor 编译期 OOM

这套约束让特征增益和训练/推理耗时之间保持了较好的平衡。

## 后续继续改特征时的建议

### 优先尝试

1. 调整 pair 白名单

只在明确知道字段语义相近时增加新的 pair 字段。建议一次只加一个序列 fid，观察线上或稳定验证集变化。

2. 调整 pair 最近窗口

可尝试：

```text
10 / 20 / 50
```

窗口太小可能漏掉兴趣延续，窗口太大可能引入过旧兴趣和更多噪声。

3. 时间特征消融

可以对比：

```text
time only
time,pair
time,item,pair
```

如果 `time only` 已明显提升，说明时间先验很强；如果 `pair` 增益明显，说明候选与历史匹配是强信号。

4. 周末深夜相关实验

当前只把周末、深夜、周末深夜作为输入特征，没有做样本加权。若后续尝试“周六周天 23/0/1 加权重”，建议作为单独实验，不要和其他大改动混在一起。

### 谨慎尝试

1. 全量 pair

不建议默认打开。全量 pair 容易跨字段 ID 空间误匹配，造成伪命中。

2. 默认启用 item target stats

除非线上结果确认有效，否则建议只作为实验分支。启用后必须保证 `feature_stats.pkl` 在推理侧同步加载。

3. 大幅增加 dense 维度

线上 baseline 每个 epoch 已经较慢，过多 dense 特征可能带来收益递减，还会增加推理压力。

4. 改 HyFormer 主干

当前提升主要来自低风险特征工程。主干结构改动的风险和排查成本更高，建议在特征实验稳定后再做。

## 每次改动后的检查清单

修改训练侧后，通常也要同步推理侧：

- `dataset.py`
- `model.py`
- `train.py`
- `Model Evaluation/dataset.py`
- `Model Evaluation/model.py`
- `Model Evaluation/infer.py`

重点检查：

- `engineered_dense_feature_names` 数量是否等于实际 `np.concatenate(parts, axis=1)` 的维度
- `_engineered_time_dim()`、`_engineered_pair_dim()` 是否和真实输出一致
- `train_config.json` 是否保存新增参数
- `infer.py` 是否从 checkpoint 恢复同一批参数
- 默认 `engineered_feature_groups` 是否训练推理一致
- 启用 item 组时是否保存并加载 `feature_stats.pkl`

推荐本地静态检查：

```bash
python -m py_compile dataset.py train.py trainer.py model.py utils.py
python -m py_compile "Model Evaluation/dataset.py" "Model Evaluation/infer.py" "Model Evaluation/model.py"
git diff --check
```

如果本地没有完整线上数据，也至少要做合成维度检查，确认：

```text
time,pair -> time 59 + pair 12 = total 71
```

## 可复用原则

- **字段语义优先于枚举数量**：少量高置信字段通常比全量弱语义字段更稳。
- **不依赖 label 的特征优先**：时间和 pair 特征比 target stats 更不容易受隐藏集漂移影响。
- **异质特征分组处理**：不同来源 dense 不要一锅端，先分组投影再融合。
- **默认配置要保守**：把高风险特征保留为可选参数，而不是默认打开。
- **训练推理必须同构**：线上训练和推理分离，任何特征维度、顺序、统计表、模型结构都必须同步。
- **每次只验证一个主要假设**：效果下降时才能定位原因，效果提升时也能沉淀经验。
