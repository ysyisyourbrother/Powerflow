# 文档索引

本目录存放项目级学习笔记。所有数值均来自当前 Python 环境的实际运行，不引用
旧格式脚本。

| 文件 | 内容 |
|---|---|
| [case9-guide.md](case9-guide.md) | IEEE 9 节点算例、数据表、母线类型、Pandapower 建模入口 |
| [case39/README.md](case39/README.md) | **IEEE 39 节点 New England 系统完整教程**（分 10 个文件，便于手机阅读） |
| [powerflow-newton.md](powerflow-newton.md) | 自研 Newton-Raphson 求解器的代码结构和逐行数学对应 |

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
| `src/powerflow/cases.py` | 读取 PYPOWER 算例、Pandapower 转换、结果回算 |
| `src/powerflow/report.py` | 迭代过程与结果表 |
| `src/examples/step1_case9.py` | 完整入门示例 |
| `src/examples/step2_pandapower.py` | Pandapower 建模仿真示例 |
| `src/examples/step3_case39.py` | IEEE 39 节点三路求解对照（PYPOWER / 自研 / Pandapower） |
| `dump_case39_tables.py` | 导出 case39 全部原始数据表 |
| `dump_case39_solution.py` | 导出 case39 全部潮流结果表 |

## 推荐顺序

1. 先读 `case9-guide.md`，明确“求解什么”和“每张表是什么”。
2. 再运行 `step1_case9.py`，亲眼看 4 次迭代。
3. 阅读 `powerflow-newton.md`，把公式、变量和 Python 代码逐项对上。
4. 读 [case39 教程](case39/README.md)，用 39 节点系统把同样的事情做一遍，并学习结果解读。
5. 进入 `powerflow-textbook/` 做系统学习和 8 个实验。

## 运行

```bash
.venv/bin/python src/examples/step1_case9.py
.venv/bin/python src/examples/step2_pandapower.py
.venv/bin/python src/examples/step3_case39.py
```
