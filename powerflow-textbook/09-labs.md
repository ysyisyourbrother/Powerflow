# 第 9 章 实验课：观察算法的每一步

## 9.0 实验目录

```text
powerflow-textbook/labs/
├── lab01_run_case9.py
├── lab02_inspect_case9.py
├── lab03_build_matrices.py
├── lab04_first_iteration.py
├── lab05_trace_newton.py
├── lab06_finite_difference.py
├── lab07_perturb_load.py
├── lab08_compare_algorithms.py
└── mismatch_from_x.py
```

所有实验都在项目根目录运行。

## 9.1 实验 1：建立基线

```bash
.venv/bin/python powerflow-textbook/labs/lab01_run_case9.py
```

观察：

- 收敛标志；
- 迭代次数；
- 总发电、总负荷和损耗；
- 最低电压母线；
- 四次迭代的范数表。

应掌握：

1. 如何判断一个潮流解可信；
2. 为什么总发电不等于总负荷；
3. 平衡发电机如何自动调节。

## 9.2 实验 2：检查数据模型

```bash
.venv/bin/python powerflow-textbook/labs/lab02_inspect_case9.py
```

观察：

```text
ref = [1]
pv  = [2 3]
pq  = [4 5 6 7 8 9]
```

检查母线表、发电机表和支路表中每一列。

思考：

- 为什么 PYPOWER 内部母线从 0 开始；
- `gen` 表的 `PG0=0` 为什么不是最终结果；
- 为什么母线 5、7、9 有负荷，但其他 PQ 母线没有。

## 9.3 实验 3：从数据到矩阵

```bash
.venv/bin/python powerflow-textbook/labs/lab03_build_matrices.py
```

观察：

- 支路 4-5 的 `y/Yff/Yft`；
- 初始复电压 `V0`；
- 计划注入 `Sbus`；
- `Ybus` 所有非零元素；
- `Ybus` 对称性。

应能手算：

$$
y=\frac1{r+jx},
$$

$$
Y_{ff}=y+j\frac b2,
\qquad
Y_{ft}=-y.
$$

## 9.4 实验 4：第一次迭代全展开

```bash
.venv/bin/python powerflow-textbook/labs/lab04_first_iteration.py
```

该实验完整打印：

1. `F0` 的 14 个元素；
2. `J0` 的 `14×14` 矩阵；
3. `dx0`；
4. 更新后的 9 个复电压；
5. 更新后的最大残差。

重点关注：

$$
\Delta\theta_2=0.171821801447,
$$

$$
\Delta\theta_3=0.088566921831,
$$

$$
\Delta V_{m,9}=-0.028543300746.
$$

## 9.5 实验 5：完整四次迭代

```bash
.venv/bin/python powerflow-textbook/labs/lab05_trace_newton.py
```

输出：

- 四次迭代的 `dx`；
- 每次更新后的 14 维 `F`；
- 与 PYPOWER 的电压误差；
- `newton_trace.csv`。

该实验对应教材第 5 章。

## 9.6 实验 6：有限差分雅可比

```bash
.venv/bin/python powerflow-textbook/labs/lab06_finite_difference.py
```

中心差分：

$$
J_{ij}^{\mathrm{fd}}
\approx
\frac{F_i(x+\epsilon e_j)-F_i(x-\epsilon e_j)}{2\epsilon}.
$$

比较：

$$
|J_{ij}^{\mathrm{analytic}}-J_{ij}^{\mathrm{fd}}|.
$$

当前最大误差约 `1.049e-09`。

把差分步长改为：

```python
step = 1e-4
step = 1e-10
```

观察截断误差和舍入误差的权衡。

## 9.7 实验 7：负荷扰动

```bash
.venv/bin/python powerflow-textbook/labs/lab07_perturb_load.py
```

母线 9 有功负荷取值：

```text
125, 150, 175, 200, 225 MW
```

当前结果：

| Pd9 MW | 迭代 | Vm9 p.u. | 发电 MW | 损耗 MW |
|---:|---:|---:|---:|---:|
| 125 | 4 | 0.957621 | 319.9547 | 4.9547 |
| 150 | 4 | 0.953587 | 345.2767 | 30.2767 |
| 175 | 4 | 0.948724 | 370.7433 | 55.7433 |
| 200 | 4 | 0.942973 | 396.3637 | 81.3637 |
| 225 | 4 | 0.936257 | 422.1500 | 107.1500 |

结论：

- 负荷增加，母线 9 电压下降；
- 发电增加；
- 线路电流增大，有功损耗快速增长。

## 9.8 实验 8：算法比较

```bash
.venv/bin/python powerflow-textbook/labs/lab08_compare_algorithms.py
```

当前结果：

| 算法 | 迭代次数 | 最低电压 | 有功损耗 |
|---|---:|---:|---:|
| NR | 4 | 0.957621 | 4.954702 |
| FDXB | 10 | 0.957621 | 4.954702 |
| GS | 271 | 0.957621 | 4.954702 |

结论：

- 三种方法给出同一个物理解；
- Newton 法利用完整雅可比，迭代次数最少；
- 快速解耦用近似矩阵，迭代次数增加；
- Gauss-Seidel 收敛最慢。

## 9.9 扩展任务

### 任务 A：改变母线 5 负荷

把：

```python
original_pd_5 = 90.0
```

改成 `120 MW`，预测电压变化。

### 任务 B：把容差放宽

把 `tol=1e-10` 改成 `1e-6`，记录迭代次数和最终残差。

### 任务 C：加入初始相角扰动

把某一 PV 母线初始相角改为 `0.05 rad`，观察是否仍收敛。

### 任务 D：比较 Pandapower

用：

```python
import pandapower as pp
pp.runpp(net, algorithm="nr")
```

比较 Pandapower 结果表与自研结果。
