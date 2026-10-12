# 潮流计算从零到 Python 求解器：基于 IEEE 9 节点算例

这是一套面向计算机科学研究生的潮流计算教材。默认读者具备线性代数、微积分、
复数、Python、NumPy 和基本数值算法基础，不要求电力系统背景。

教材主线：

1. 用 PYPOWER 读取 IEEE 9 节点数据；
2. 用 Pandapower 建立同一网络的工程模型；
3. 自研一个透明 Python 牛顿-拉夫逊求解器；
4. 把每个公式与 Python 语句逐项对照；
5. 完整展示 case9 的四次迭代；
6. 回算发电机出力、线路潮流和网络损耗；
7. 用成熟库和有限差分验证自研结果。

## 学完后应能完成

- 解释 PQ、PV、平衡母线的已知量和未知量；
- 从支路参数构造节点导纳矩阵 `Ybus`；
- 从母线/发电机数据构造计划功率注入 `Sbus`；
- 写出 14 维状态向量、14 维残差和 `14×14` 雅可比；
- 独立完成一次极坐标牛顿迭代；
- 读懂 `src/powerflow/newton_raphson.py`；
- 用 PYPOWER 验证数值结果；
- 用 Pandapower 建立并运行仿真模型；
- 为 AI 潮流计算比赛扩展自己的求解器。

## 章节目录

1. [学习地图](00-learning-map.md)
2. [潮流问题与 IEEE 9 节点系统](01-background-and-case9.md)
3. [Python 数据模型、标幺值与母线类型](02-data-model-and-per-unit.md)
4. [节点导纳矩阵与网络方程](03-ybus-network-equations.md)
5. [功率平衡、残差与未知向量](04-power-balance-and-unknowns.md)
6. [牛顿-拉夫逊法与完整迭代过程](05-newton-raphson.md)
7. [Python 代码与公式逐步对照](06-newton-code-walkthrough.md)
8. [雅可比矩阵：推导、Python 实现与验证](07-jacobian.md)
9. [结果回算、线路潮流与物理校验](08-solution-recovery.md)
10. [实验课：观察算法的每一步](09-labs.md)
11. [从 case9 走向比赛求解器](10-competition-extension.md)
12. [附录 A：Python / PYPOWER / Pandapower 速查](appendix-a-python-cheatsheet.md)
13. [附录 B：源码地图](appendix-b-source-map.md)
14. [附录 C：习题与参考答案](appendix-c-exercises.md)

## 环境

当前版本：

| 包 | 版本 |
|---|---:|
| Python | 3.12 |
| PYPOWER | 5.1.21 |
| Pandapower | 3.5.6 |
| NumPy | 2.4.6 |
| SciPy | 1.18.1 |

安装：

```bash
./install.sh
```

自检：

```bash
.venv/bin/python src/examples/smoke_test.py
```

## 运行实验

每章末尾会给出对应实验。所有实验位于 `powerflow-textbook/labs/`：

```bash
.venv/bin/python powerflow-textbook/labs/lab01_run_case9.py
.venv/bin/python powerflow-textbook/labs/lab02_inspect_case9.py
.venv/bin/python powerflow-textbook/labs/lab03_build_matrices.py
.venv/bin/python powerflow-textbook/labs/lab04_first_iteration.py
.venv/bin/python powerflow-textbook/labs/lab05_trace_newton.py
.venv/bin/python powerflow-textbook/labs/lab06_finite_difference.py
.venv/bin/python powerflow-textbook/labs/lab07_perturb_load.py
.venv/bin/python powerflow-textbook/labs/lab08_compare_algorithms.py
```

## 代码约定

- PYPOWER 内部使用 0 基索引；教材展示母线编号时转换为 `1..9`。
- `Vm` 表示电压幅值，`Va` 表示电压相角。
- PYPOWER 的 `Va` 在数据表和输出中通常是度，核心算法内部使用弧度。
- `Sbus` 表示“发电机减负荷”，不是“负荷减发电机”。
- 功率核心计算使用 p.u.；输入输出通常使用 MW/Mvar。
- 平衡节点的 `P,Q` 与 PV 节点的 `Q` 都是求解后的结果。
