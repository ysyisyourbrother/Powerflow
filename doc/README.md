# 文档索引

本目录存放项目级学习笔记。所有数值均来自当前 Python 环境的实际运行，
代码示例只引用当前保留的实现。

> 数据集相关的说明、教程和统计图已统一放到 [`../dataset/`](../dataset/README.md)，
> 本目录只保留 IEEE 39 数据讲解和潮流通用教材。

## 从哪里开始

| 你的目标 | 先读 |
|---|---|
| 搞清楚比赛数据怎么组织、字段和单位是什么 | [dataset/README.md](../dataset/README.md) |
| 系统学习从算例到训练样本的全过程 | [competition-dataset-guide.md](../dataset/competition-dataset-guide.md) |
| 从零理解 IEEE 39 节点系统 | [case39/README.md](case39/README.md) |
| 系统学习潮流计算与牛顿法 | [powerflow-textbook/README.md](powerflow-textbook/README.md) |
| 直接看数据集字段和生成命令 | [../dataset/ieee39/processed/competition_v1/README.md](../dataset/ieee39/processed/competition_v1/README.md) |

## 主要文档

| 文件 | 内容 |
|---|---|
| [../dataset/README.md](../dataset/README.md) | **数据集总入口**：数据布局、字段约定、生成方式；详细字段表见 `competition_v1/README.md` |
| [../dataset/competition-dataset-guide.md](../dataset/competition-dataset-guide.md) | **数据集完整教程**：从 IEEE 39 算例到字段、掩码、打标、读取和训练建议 |
| [../dataset/model-design.md](../dataset/model-design.md) | **神经网络模型设计**：输入构建、数据处理、架构、损失和训练方案（迭代中） |
| [case39/README.md](case39/README.md) | **IEEE 39 节点 New England 系统完整教程**（分 10 个文件，便于手机阅读） |
| [powerflow-textbook/README.md](powerflow-textbook/README.md) | 从数学、标幺值、Ybus 到牛顿法和代码实现的系统教材 |

## IEEE 39 节点教程的分册

`case39/` 目录把完整教程拆成了 10 个小节，从上到下按顺序阅读：

| # | 文件 | 内容 |
|---:|---|---|
| — | [README.md](case39/README.md) | 目录导航 + 快速开始 + 关键数字速查 |
| 1 | [01-data-overview.md](case39/01-data-overview.md) | 系统背景、数据来源、bus/gen/branch 逐列讲解 |
| 2 | [02-data-tables.md](case39/02-data-tables.md) | 39 母线 / 10 发电机 / 46 支路完整数据表 |
| 3 | [03-bus-types-pitfalls.md](case39/03-bus-types-pitfalls.md) | PQ/PV/平衡节点、Ybus 手工算例、6 个数据陷阱 |
| 4 | [04-newton-theory.md](case39/04-newton-theory.md) | 潮流方程、极坐标功率方程、雅可比、算法流程 |
| 5 | [05-solving-practice.md](case39/05-solving-practice.md) | PYPOWER 求解、平启动二次收敛、交叉验证 |
| 6 | [06-results-voltage.md](case39/06-results-voltage.md) | 母线电压与发电机出力的解读 |
| 7 | [07-results-flow.md](case39/07-results-flow.md) | 支路潮流、功率平衡、结果图、自检清单 |
| 8 | [08-code.md](case39/08-code.md) | 配套脚本、运行命令、最小可运行示例 |
| 9 | [09-faq.md](case39/09-faq.md) | 10 个常见问题与排错 |
| 10 | [10-next-steps.md](case39/10-next-steps.md) | 巩固实验、配套教材、深入方向 |

## 配套代码

| 代码 | 作用 |
|---|---|
| `src/powerflow/newton_raphson.py` | 残差、功率方程、解析雅可比、迭代主循环 |
| `src/powerflow/cases.py` | 读取 PYPOWER 算例、结果回算 |
| `src/powerflow/dataset/competition.py` | 赛事格式 schema、掩码、打标、读写 |
| `src/examples/build_competition_dataset.py` | 生成 2000 条赛事对齐样本 |
| `src/examples/step3_case39.py` | IEEE 39 节点两路求解对照（PYPOWER / 自研） |
| `doc/case39/dump_case39_tables.py` | 导出 case39 全部原始数据表 |
| `doc/case39/dump_case39_solution.py` | 导出 case39 全部潮流结果表 |

## 推荐顺序

1. 先读 [dataset/README.md](../dataset/README.md)，明确数据布局和字段约定；
   字段速查见 [competition_v1/README.md](../dataset/ieee39/processed/competition_v1/README.md)。
2. 再读 [case39/README.md](case39/README.md)，把 39 节点系统的数据、母线类型、
   潮流结果逐项弄懂。
3. 需要补理论时，读 [powerflow-textbook/](powerflow-textbook/README.md)：
   标幺值 → Ybus → 功率方程 → 牛顿法 → 代码实现。
4. 想动手时，运行 `src/examples/build_competition_dataset.py` 和
   `src/examples/step3_case39.py`，再看 `tests/` 里的断言。

## 运行

当前环境使用 conda 的 `powerflow` 环境（Python 3.12）。如果你的环境已激活，
直接用 `python`；如果使用项目虚拟环境，把 `python` 换成 `.venv/bin/python`。

```bash
python src/examples/step3_case39.py
python src/examples/build_competition_dataset.py --n-samples 2000
python -m pytest -q
```
