# Home Credit Default Risk 实验报告

Yitong Bai yb2636 || repo:https://github.com/catRunnerCN/Applied-ML/tree/main/A2

## Introduction

本实验研究 Kaggle **Home Credit Default Risk** 的二分类任务，目标是判断贷款申请人是否更可能出现还款困难，其中 `TARGET=1` 表示风险较高，`TARGET=0` 表示风险较低。实验只使用主表 `application_train.csv`，不与其他表进行合并，这样可以把重点放在 **XGBoost** 和 **MLPClassifier** 两种模型的比较上。原始数据共有 **307,511** 条样本和 **122** 列。删除 ID 列并加入少量特征工程后，最终使用 **124** 个特征，其中 **108** 个是数值特征，**16** 个是类别特征。由于违约样本只占 **8.07%**，而非违约样本占 **91.93%**，数据明显不平衡，所以本实验把 **AUC-PR** 作为最重要的指标，同时也报告 Precision、Recall、F1 和 Accuracy。

## Methods

数据采用分层抽样，划分为 **Train/Validation/Test = 70%/15%/15%**，对应 **215,257 / 46,127 / 46,127** 条样本。验证集用于调参，测试集只在最后使用一次。所有缺失值填补、标准化和编码操作都只在训练集上完成，然后再应用到验证集和测试集，这样可以避免数据泄漏。预处理阶段删除 `SK_ID_CURR`，并把异常值 `DAYS_EMPLOYED=365243`（共 **55,374** 次）当作缺失值处理。缺失很多的特征，如 `COMMONAREA_*`、`NONLIVINGAPARTMENTS_*`、`FLOORSMIN_*` 和 `YEARS_BUILD_*`，仍然保留。特征工程只加入四个比较容易理解的变量：`CREDIT_INCOME_RATIO`、`ANNUITY_INCOME_RATIO`、`INCOME_PER_FAMILY_MEMBER` 和 `EXT_SOURCE_MEAN`。XGBoost 使用数值中位数填补、类别众数填补和 one-hot 编码，不做标准化；MLP 在这些步骤之外，还对数值特征做标准化，并把输入转成 dense 格式。

## Results

### XGBoost

XGBoost 的 baseline 设置为 `n_estimators=1000`、`learning_rate=0.05`、`max_depth=5`、`subsample=0.8`、`colsample_bytree=0.8`。在验证集上，结果为 **Accuracy 0.9195、Precision 0.5476、Recall 0.0185、F1 0.0358、AUC-PR 0.2465**。这说明模型整体比较保守，不太容易把样本判成正类，但一旦判成正类，往往更可靠。logloss 曲线也说明训练过程比较平稳，不过到后期提升已经很有限，并且出现了轻微过拟合。

![Figure 1: XGBoost train/validation logloss](/figures/xgb_train_val_logloss.png)

| Learning Rate | Validation F1 | Validation AUC-PR | Boosting Rounds |
|---|---:|---:|---:|
| 0.01 | 0.0263 | 0.2469 | 1000 |
| 0.10 | 0.0429 | 0.2467 | 193 |
| 0.30 | 0.0395 | 0.2414 | 36 |

![Figure 2: XGBoost learning rate comparison](/figures/xgb_learning_rate_comparison.png)

如果以 AUC-PR 为主，`0.01` 的结果略好一些。继续比较树深和采样率后可以看到，树更深时效果更好，而 `subsample` 取中间值通常比 `1.0` 更稳定。

| max_depth | subsample | F1 | AUC-PR |
|---:|---:|---:|---:|
| 3 | 0.6 | 0.0138 | 0.2413 |
| 3 | 0.8 | 0.0123 | 0.2399 |
| 3 | 1.0 | 0.0096 | 0.2383 |
| 5 | 0.6 | 0.0305 | 0.2472 |
| 5 | 0.8 | 0.0263 | 0.2469 |
| 5 | 1.0 | 0.0211 | 0.2457 |
| 7 | 0.6 | 0.0359 | 0.2493 |
| 7 | 0.8 | 0.0339 | 0.2494 |
| 7 | 1.0 | 0.0304 | 0.2486 |

| reg_alpha | reg_lambda | Validation F1 | Validation AUC-PR |
|---:|---:|---:|---:|
| 0.0 | 1.0 | 0.0339 | 0.2494 |
| 0.0 | 5.0 | 0.0334 | 0.2489 |
| 0.0 | 10.0 | 0.0339 | 0.2488 |
| 0.1 | 1.0 | 0.0349 | 0.2490 |
| 0.1 | 5.0 | 0.0344 | 0.2494 |
| 0.1 | 10.0 | 0.0329 | 0.2485 |
| 1.0 | 1.0 | 0.0334 | 0.2493 |
| 1.0 | 5.0 | 0.0334 | 0.2491 |
| 1.0 | 10.0 | 0.0319 | 0.2482 |

最终采用 `learning_rate=0.01, max_depth=7, subsample=0.8, reg_alpha=0.1, reg_lambda=5.0`。这个配置在验证集上的结果为 **Accuracy 0.9198、Precision 0.6055、Recall 0.0177、F1 0.0344、AUC-PR 0.2494**。从特征重要性来看，`EXT_SOURCE_MEAN` 最关键，后面依次是 `EXT_SOURCE_3`、`EXT_SOURCE_2`、`EXT_SOURCE_1`、`NAME_INCOME_TYPE_Pensioner`、`NAME_EDUCATION_TYPE_Higher education`、`CODE_GENDER_M`、`FLAG_DOCUMENT_3` 和 `AMT_GOODS_PRICE`。

![Figure 3: XGBoost feature importance](./figures/xgb_feature_importance.png)

### MLP

MLP 的 baseline 设置为 `hidden_layer_sizes=(128,64)`、`activation=relu`、`learning_rate_init=0.001`、`max_iter=50`。在验证集上，结果为 **Accuracy 0.8955、Precision 0.2132、Recall 0.1093、F1 0.1445、AUC-PR 0.1466**。和 XGBoost 相比，MLP 更容易把样本判成正类，所以 Recall 更高，但同时误报也更多，因此 Precision 和 AUC-PR 都更低。

| Hidden Layers | Validation F1 | Validation AUC-PR | Train Time (s) |
|---|---:|---:|---:|
| `(64,)` | 0.0977 | 0.1984 | 29.77 |
| `(128, 64)` | 0.1445 | 0.1466 | 54.59 |
| `(256, 128, 64)` | 0.1577 | 0.1294 | 99.31 |

![Figure 4: MLP architecture comparison](./figures/mlp_architecture_comparison.png)

更深的网络虽然让 F1 上升，但 AUC-PR 反而下降，所以如果把 AUC-PR 作为主指标，`(64,)` 更合适。激活函数比较中，`relu` 的 **F1=0.0977、AUC-PR=0.1984**，`tanh` 的 **F1=0.0992、AUC-PR=0.1887**，因此最终保留 `relu`。学习率和训练轮数的比较结果如下。

| Learning Rate | Validation F1 | Validation AUC-PR | n_iter |
|---:|---:|---:|---:|
| 0.001 | 0.0977 | 0.1984 | 50 |
| 0.01 | 0.0872 | 0.1836 | 50 |
| 0.1 | 0.0005 | 0.1899 | 14 |

| max_iter | Validation F1 | Validation AUC-PR | Train Time (s) |
|---:|---:|---:|---:|
| 30 | 0.0956 | 0.2035 | 20.60 |
| 50 | 0.0977 | 0.1984 | 35.62 |
| 80 | 0.1028 | 0.1849 | 56.36 |

最终采用 `hidden_layer_sizes=(64,), activation=relu, learning_rate_init=0.001, max_iter=30`。这个配置在验证集上的结果为 **Accuracy 0.9164、Precision 0.3764、Recall 0.0548、F1 0.0956、AUC-PR 0.2035**。loss curve 显示，训练损失继续下降，但验证集的 AUC-PR 没有继续变好，说明训练更久并不一定带来更好的测试表现。

![Figure 5: MLP training loss curve](./figures/mlp_loss_curve.png)

## GBDT vs MLP Comparison

最终测试集结果如下：

| Model | Accuracy | Precision | Recall | F1 | AUC-PR |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.9198 | 0.6000 | 0.0185 | 0.0359 | 0.2460 |
| MLP | 0.9169 | 0.3930 | 0.0542 | 0.0953 | 0.2119 |

| Model | Best Config | Final Train Time (s) |
|---|---|---:|
| XGBoost | `lr=0.01, depth=7, subsample=0.8, reg_alpha=0.1, reg_lambda=5.0` | 22.35 |
| MLP | `(64,), relu, lr=0.001, max_iter=30` | 25.92 |

![Figure 6: Final metrics comparison](./figures/final_metrics_comparison.png)

从最终结果看，XGBoost 在 **AUC-PR 0.2460**、**Precision 0.6000** 和 Accuracy 上都更好，说明它更擅长把真正高风险的客户排在前面，也更适合在默认阈值下给出较可靠的正类判断。MLP 的优势在于 **Recall 0.0542** 和 **F1 0.0953**，说明它能找出更多违约样本，但同时也会带来更多误报。对于这类正负样本很不平衡的任务，AUC-PR 比单纯的 Accuracy 更重要，因此 XGBoost 更适合作为主模型，MLP 更适合作为对照模型。

## Discussion

整体结果符合表格数据建模中的常见经验：GBDT 通常更适合处理这种同时包含数值、类别和缺失值的结构化数据。XGBoost 的 Recall 很低，更多反映的是默认分类阈值比较保守，而不是模型不会排序。如果后续继续做阈值调整，Recall 和 F1 还有提升空间。MLP 的结果说明，更深的网络和更长的训练时间虽然有时能提高固定阈值下的 F1，但整体排序能力并没有同步变好，甚至会下降。`EXT_SOURCE_MEAN` 排名第一也说明，简单但有业务意义的特征工程仍然很有价值。本实验也有一些限制，例如只使用了主表，没有做多表信息融合，没有进行阈值调优，也没有使用 class weight 或重采样方法；另外，MLP 使用的是 sklearn 版本，模型灵活性也比较有限。从 bias-variance 的角度看，XGBoost 在本实验设置下取得了更均衡的结果。

## AI Tool Disclosure

本报告在写作整理阶段使用了 ChatGPT，主要用于压缩原始长报告、调整结构和润色语言。实验设计、数据清洗、特征工程、XGBoost 与 MLP 的训练和调参等部分先由AI进行教学，并给我一个代码框架，最后由我实现框架，与AI一起Debug。结果分析以及最终结论都由本人完成。AI 工具只参与了教学，表达和整理，不替代实验和代码本身。
