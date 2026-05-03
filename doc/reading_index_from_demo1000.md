# 文档总索引（demo1000 同构版本）

这页用于把你现有的 4 份核心文档串成一条“最快理解项目”的阅读路径。

---

## 1. 推荐阅读顺序（强烈建议按这个来）

1. `dataset_pipeline_from_demo1000.md`  
2. `train_pipeline_from_demo1000.md`  
3. `model_pipeline_from_demo1000.md`  
4. `trainer_pipeline_from_demo1000.md`  

理由：

- 先搞懂输入数据怎么变成 batch（dataset）
- 再搞懂主流程怎么把模块拼起来（train）
- 再看模型内部前向（model）
- 最后看训练执行与评估保存（trainer）

---

## 2. 每份文档建议重点看哪几节

## 2.1 `dataset_pipeline_from_demo1000.md`

优先看：

1. **关键变量先看懂**  
2. **主流程图（总览 + 详细）**  
3. **完整样例（真实列名族 + 拼接写入）**  
4. **最终 batch 字段一览**  

读完后你应该能回答：

- 每个原始列最终落在 batch 的哪个字段？
- `offset/length` 写入是怎么做的？
- `seq_d / seq_d_len / seq_d_time_bucket` 三者关系是什么？

## 2.2 `train_pipeline_from_demo1000.md`

优先看：

1. **详细流程图（并行/汇合）**  
2. **步骤 4~7（分组、specs、建模、启动训练）**  
3. **完整样例（shape 代入）**  

读完后你应该能回答：

- `ns_groups` 与 `feature_specs` 是怎么从数据元信息变来的？
- `model_args` 的关键来源有哪些？
- `train.py` 到底负责“计算”还是“编排”？

## 2.3 `model_pipeline_from_demo1000.md`

优先看：

1. **输入输出协议**  
2. **详细流程图（NS分支 + 序列分支 + HyFormer融合）**  
3. **完整样例（shape 代入）**  

读完后你应该能回答：

- `ModelInput` 到 `logits` 的每一步 shape 怎么变？
- `Nq / Nns / T / D` 是怎么关联的？
- `rank_mixer_mode=full` 的约束为何出现？

## 2.4 `trainer_pipeline_from_demo1000.md`

优先看：

1. **详细流程图（train step + evaluate + checkpoint）**  
2. **步骤 4~7（loss更新、验证、最佳保存、稀疏重启）**  
3. **完整样例（train/eval shape 链路）**  

读完后你应该能回答：

- 一个训练 step 里具体做了哪些张量操作？
- AUC/logloss 是怎么计算并触发 early stopping 的？
- 为什么 checkpoint 保存逻辑要分“可能最佳/确认最佳”两步？

---

## 3. 一条“30分钟速读路径”

如果你时间紧，按下面看：

1. 先看 `dataset` 的“完整样例”和“最终 batch 字段”  
2. 再看 `model` 的“输入输出协议 + 完整样例”  
3. 再看 `trainer` 的“_train_step shape 链路”  
4. 最后看 `train` 的“详细流程图”补全全局认知  

这样能最快建立“数据 -> 模型 -> 训练”的闭环理解。

---

## 4. 对照源码时的入口文件

当你看完文档想回到源码验证，建议按这个顺序点开：

1. `dataset.py`（先看 `_load_schema`、`_convert_batch`）
2. `model.py`（先看 `forward`，再看子模块）
3. `trainer.py`（先看 `_train_step`、`evaluate`）
4. `train.py`（最后看组装与启动）

---

## 5. 一句话总结

这套文档的最佳使用方式是：**先用 `dataset` 建立输入语义，再用 `model` 建立前向语义，最后用 `trainer+train` 建立执行语义**。

