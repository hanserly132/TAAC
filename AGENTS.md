# AGENTS.md

## 版本说明

当前版本可视为：**baseline_v1 + 稳健 feature engineering v2 增强版**。

本仓库最初是 TAAC2026 推荐系统/PCVR 预测任务的 baseline。当前改动重点不是重写模型结构，而是在保持 baseline 训练、推理格式兼容的前提下，增加可控的工程特征：

- 默认启用增强时间特征和语义白名单 pair 特征：`time,pair`
- item 侧 target/stat 统计特征保留为可选实验，默认不启用
- 启用 item 组时训练阶段保存 `feature_stats.pkl`
- 启用 item 组时推理阶段加载同一份 `feature_stats.pkl`，避免训练/推理特征语义不一致
- `DenseGroupFusion` 会按 raw dense、time、pair、item 组分别投影后门控融合
- 默认关闭 `torch.compile`，规避线上平台 inductor 编译阶段 CUDA OOM 风险
- 该版本线上效果相较 baseline 有明显提升，经验复盘见 `doc/feature_engineering_v2_lessons.md`
- 后续改特征时应同步参考外部 schema 页面：`https://puiching-memory.github.io/TAAC_2026/analysis/feature-schema/`

一句话概括：**这是一个在原 PCVRHyFormer baseline 上追加增强时间特征、语义白名单 pair 特征和异质 dense 分组投影，并保持线上训练/推理格式兼容的版本。**

## 项目目标

项目面向 TAAC2026 推荐系统比赛，任务是对用户-候选 item 交互进行转化概率预测，即 PCVR / post-click conversion rate prediction。

输入数据是 parquet 格式的扁平列布局，样例位于 `data_sample_1000/demo_1000.parquet`。样例数据约 1000 行、120 列，主要包含：

- 基础 ID 与标签：`user_id`、`item_id`、`label_type`、`label_time`、`timestamp`
- 用户离散特征：`user_int_feats_*`
- 用户 dense 特征：`user_dense_feats_*`
- item 离散特征：`item_int_feats_*`
- 四个行为序列域：`domain_a_seq_*`、`domain_b_seq_*`、`domain_c_seq_*`、`domain_d_seq_*`

训练目标是二分类转化预测，输出 logits，训练时使用 BCE 或 focal loss，评估指标包含 AUC 和 logloss。

## 主要文件职责

### 训练侧

- `run.sh`
  - 线上训练任务的强制入口。
  - 设置 `PYTHONPATH` 后调用 `train.py`。
  - 当前默认参数包含 `rankmixer` NS tokenizer、`num_epochs=10`、`batch_size=512`、`seq_max_lens="seq_a:512,seq_b:512,seq_c:1024,seq_d:1024"`。

- `train.py`
  - 训练主控脚本，负责读取平台环境变量、构建 dataloader、构建模型、初始化 trainer。
  - 重要环境变量：`TRAIN_DATA_PATH`、`TRAIN_CKPT_PATH`、`TRAIN_LOG_PATH`、`TRAIN_TF_EVENTS_PATH`。
  - 工程特征参数：
    - `--use_engineered_features` / `--no_engineered_features`
    - `--engineered_feature_groups`，默认 `time,pair`
    - `--stats_smoothing`，默认 `20.0`
    - `--pair_recent_steps`，默认 `20`
    - `--pair_seq_fids`，默认 `seq_a:38,seq_b:69,seq_c:47,seq_d:23`
    - `--pair_candidate_fids`，默认 `item_id,11`
    - `--dense_projection_mode`，默认 `group_fusion`
  - `--compile_model` 默认关闭；只有显式传入才启用 `torch.compile`。

- `dataset.py`
  - 数据读取与 batch 构造核心。
  - 主要类/函数：`FeatureSchema`、`PCVRParquetDataset`、`get_pcvr_data`、`build_feature_stats`、`save_feature_stats`、`load_feature_stats`。
  - 使用 parquet row group 划分 train/valid。
  - 将原始 flat columns 转成模型输入需要的：
    - `user_int_feats`
    - `item_int_feats`
    - `user_dense_feats`
    - `item_dense_feats`
    - `seq_*`
    - `seq_*_len`
    - `seq_*_time_bucket`
  - 新增工程 dense 特征会追加到 `user_dense_feats` 尾部，使用内部 fid `ENGINEERED_DENSE_FID = 100000`。
  - 同时提供 `user_dense_group_indices`，供模型按 `pretrain_dense / stat_dense / time_engineered / pair_engineered / item_engineered` 做异质分组投影。

- `model.py`
  - PCVRHyFormer 模型主体。
  - 输入协议由 `ModelInput` 定义，包括 user/item 离散特征、dense 特征、多域序列、序列长度、时间桶。
  - 主模型为 `PCVRHyFormer`，核心结构包含：
    - user/item NS tokenizer
    - 多域序列 embedding
    - time bucket embedding
    - MultiSeqQueryGenerator
    - MultiSeqHyFormerBlock
    - RankMixerBlock
    - DenseGroupFusion
    - 最终 MLP 输出 logits
  - 当前默认 NS tokenizer 使用 `rankmixer`，可降低 NS token 数量，控制 `d_model % T` 约束。

- `trainer.py`
  - 训练循环、验证、early stopping、checkpoint 保存。
  - sparse embedding 参数使用 Adagrad，dense 参数使用 AdamW。
  - checkpoint 会保存到 `global_step...` 开头的目录，满足平台识别要求。
  - checkpoint sidecar 包含 `schema.json`、`train_config.json`、`ns_groups.json` 以及启用 item 统计特征时的 `feature_stats.pkl`。

### 推理侧

- `Model Evaluation/infer.py`
  - 线上推理入口，必须包含无参数 `main()`。
  - 读取平台环境变量：
    - `MODEL_OUTPUT_PATH`
    - `EVAL_DATA_PATH`
    - `EVAL_RESULT_PATH`
  - 自动解析 checkpoint 目录和 `model.pt`。
  - 优先使用 checkpoint 中的 `schema.json` 和 `train_config.json` 来恢复训练时结构。
  - 当启用工程特征且包含 item 统计特征时，会从 checkpoint 目录加载 `feature_stats.pkl`。
  - 输出必须为 `EVAL_RESULT_PATH/predictions.json`，格式为：
    ```json
    {
      "predictions": {
        "user_id": 0.1234
      }
    }
    ```

- `Model Evaluation/dataset.py`、`Model Evaluation/model.py`
  - 与训练侧同步的推理依赖文件。
  - 修改训练侧 `dataset.py` 或 `model.py` 时，通常也需要同步到 `Model Evaluation/`，否则推理可能因 schema、特征维度或模型结构不一致而失败。

## 当前特征工程改动

工程特征默认追加到 `user_dense_feats`，这样不改变原始 user/item int 特征、item dense 特征和序列输入结构，也不会额外增加 NS token 数。

### 1. `time` 序列时间特征

针对 `seq_a`、`seq_b`、`seq_c`、`seq_d` 四个行为序列域构造：

- 序列长度
- 最近一次行为距当前样本时间的 recency
- 序列时间跨度
- 行为间隔的 mean / min / max
- 最近 `1h / 6h / 1d / 3d / 7d / 30d` 行为计数
- 样本 timestamp 的 hour/day 周期特征
- 周末、深夜 `23/0/1`、周末深夜二值特征

这类特征只依赖样本自身，不依赖 label，因此训练和推理都可直接计算。当前版本只用 `0 < seq_ts <= timestamp` 的历史行为计算 recency、span、gap 和窗口计数。

### 2. `item` 侧统计特征

基于训练集 row group 构建 train-only item 统计表：

- `item_id` 曝光次数
- `item_id` 正样本次数
- `item_id` 平滑 CVR
- 各 `item_int_feats_*` 取值的曝光次数、正样本次数、平滑 CVR

训练样本使用 leave-one-out 调整，降低标签泄漏风险。统计表保存在 checkpoint 目录的 `feature_stats.pkl` 中，推理时必须加载同一份统计表。该组由于存在隐藏测试集分布漂移和 target encoding 过拟合风险，当前默认不启用。

### 3. `pair` 匹配特征

当前默认启用。默认配置：

```bash
--engineered_feature_groups time,pair
--pair_recent_steps 20
--pair_seq_fids "seq_a:38,seq_b:69,seq_c:47,seq_d:23"
--pair_candidate_fids "item_id,11"
```

pair 特征会计算候选 item 信息与用户最近历史序列的命中关系，包括是否命中、命中次数、最近一次命中的时间差。当前只匹配 schema 分析中较像 ID 的序列通道，避免全量 sideinfo 跨字段伪命中。

### 4. Dense 异质分组投影

`model.py` 中的 `DenseGroupFusion` 会把 `user_dense_feats` 按来源切分为多个组，每组独立 `Linear + LayerNorm + SiLU`，再用门控融合成 1 个 dense token。这样不增加 NS token 数，不破坏 `d_model % T`。

## 训练/推理一致性要求

本项目训练和推理阶段是分开的，因此下列文件/配置必须保持一致：

- `schema.json`
- `train_config.json`
- `ns_groups.json`
- `feature_stats.pkl`
- `dataset.py` 中工程特征构造逻辑
- `model.py` 中模型结构和默认参数解释

尤其要注意：`feature_stats.pkl` 是训练阶段从训练集构建出的 item 统计表。只有显式启用 `item` 组时才会生成和加载；推理阶段无法重新得到完整训练集统计，因此启用 item 组时必须从 checkpoint 加载，才能保证 item 统计特征的含义和维度与训练阶段一致。

## 平台约束

训练阶段：

- 平台自动执行 `run.sh`。
- 模型权重必须保存到 `TRAIN_CKPT_PATH`。
- checkpoint 子目录必须以 `global_step` 开头。
- TensorBoard 标量日志可写入 `TRAIN_TF_EVENTS_PATH`。

推理阶段：

- 上传文件必须包含 `infer.py`。
- `infer.py` 必须定义无参数 `main()`。
- 推理脚本总大小限制为 100 MB。
- 单次推理任务限制 30 分钟。
- 必须生成 `EVAL_RESULT_PATH/predictions.json`。
- `predictions.json` 的 key 必须是测试集中的有效 `user_id`，value 是 0 到 1 的转化概率。

## 效率与风险记录

- 线上 baseline 约每个 epoch 1 小时，工程特征不能无限增加维度。
- 当前默认只启用 `time,pair`，约新增 71 维工程 dense 特征。
- `pair` 特征只使用白名单序列 fid 和最近 20 步窗口，主要是为了控制训练和 30 分钟推理时间。
- 曾遇到线上 `torch.compile` / inductor 编译阶段 CUDA OOM，因此当前默认 eager 模式；如需尝试编译，必须显式传 `--compile_model`。
- `feature_stats.pkl` 不应提交到 git，它属于训练产物，已在 `.gitignore` 中忽略。
- `model.pt`、`*.pth`、`*.ckpt`、`ckpt/`、`events/`、`logs/` 等训练产物也已忽略。

## 推荐阅读顺序

如果后续继续维护该项目，建议按下面顺序理解：

1. `data_sample_1000/README.md`：先了解数据列结构。
2. `doc/reading_index_from_demo1000.md`：查看已有文档索引。
3. `https://puiching-memory.github.io/TAAC_2026/analysis/feature-schema/`：查看字段语义、基数、multi-hot 维度和 train/eval/demo/infer schema 差异。
4. `doc/feature_engineering_v2_lessons.md`：理解当前有效特征工程的经验、原理和后续改动原则。
5. `dataset.py`：理解 parquet 到 batch 的转换，以及工程特征追加逻辑。
6. `train.py`：理解训练入口、参数、环境变量和模型组装。
7. `model.py`：理解 `ModelInput -> logits` 的前向链路。
8. `trainer.py`：理解训练 step、验证、checkpoint 保存。
9. `Model Evaluation/infer.py`：理解线上推理如何恢复训练配置并输出 `predictions.json`。

## 常用命令

本地训练入口示例：

```bash
bash run.sh
```

关闭工程特征：

```bash
bash run.sh --no_engineered_features
```

显式启用 item 统计特征：

```bash
bash run.sh --engineered_feature_groups time,item,pair
```

显式启用 `torch.compile`：

```bash
bash run.sh --compile_model
```

检查 git 状态：

```bash
git status --short --branch
```

## GitHub 状态

当前项目已初始化为 git 仓库，并推送到：

```text
https://github.com/hanserly132/TAAC.git
```

初始提交：

```text
e3b0f82 Initial feature-engineering baseline
```
