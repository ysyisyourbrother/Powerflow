# IEEE 9 节点算例详解

本篇回答四个问题：

1. IEEE 9 节点系统是什么？
2. PYPOWER 的数据表如何描述电网？
3. 母线为什么分为 PQ、PV 和平衡节点？
4. 同一算例如何转成 Pandapower 网络？

---

## 1. 系统规模

当前项目调用：

```python
from pypower.api import case9

ppc = case9()
```

得到：

| 元素 | 数量 |
|---|---:|
| 母线 bus | 9 |
| 发电机 gen | 3 |
| 支路 branch | 9 |
| 有功负荷 | 315 MW |
| 无功负荷 | 115 Mvar |
| 基准容量 | 100 MVA |

三个发电机分别接在母线 1、2、3。负荷接在母线 5、7、9：

| 母线 | 有功负荷 | 无功负荷 |
|---:|---:|---:|
| 5 | 90 MW | 30 Mvar |
| 7 | 100 MW | 35 Mvar |
| 9 | 125 MW | 50 Mvar |

节点类型：

| 类型码 | 类型 | 已知量 | 未知量 | 当前数量 |
|---:|---|---|---|---:|
| 1 | PQ 母线 | `P,Q` | `Vm,θ` | 6 |
| 2 | PV 母线 | `P,Vm` | `Q,θ` | 2 |
| 3 | 平衡母线 | `Vm,θ` | `P,Q` | 1 |

在 PYPOWER 内部连续编号后：

```text
ref = [1]
pv  = [2, 3]
pq  = [4, 5, 6, 7, 8, 9]
```

这里的编号是 Python 的 `0` 基内部索引；教材展示结果时统一加 1，恢复为母线
编号 `1..9`。

---

## 2. PYPOWER 数据结构

### 2.1 母线表

最常用的列：

| 常量 | 含义 |
|---|---|
| `BUS_I` | 母线编号 |
| `BUS_TYPE` | 1=PQ，2=PV，3=平衡 |
| `PD,QD` | 有功、无功负荷，MW/Mvar |
| `GS,BS` | 并联电导、电纳 |
| `VM,VA` | 电压幅值、相角初值，p.u./度 |
| `BASE_KV` | 基准电压 |

打印：

```python
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VM, VA

for row in ppc["bus"]:
    print(row[BUS_I], row[BUS_TYPE], row[PD], row[QD], row[VM], row[VA])
```

### 2.2 发电机表

最常用的列：

| 常量 | 含义 |
|---|---|
| `GEN_BUS` | 发电机接入母线 |
| `PG,QG` | 有功、无功初值，MW/Mvar |
| `QMAX,QMIN` | 无功上下限 |
| `VG` | 电压幅值设定 |
| `GEN_STATUS` | 是否在运行 |
| `PMAX,PMIN` | 有功上下限 |

### 2.3 支路表

最常用的列：

| 常量 | 含义 |
|---|---|
| `F_BUS,T_BUS` | 首端、末端母线 |
| `BR_R,BR_X` | 串联电阻、电抗，p.u. |
| `BR_B` | 总充电电纳，p.u. |
| `TAP,SHIFT` | 变压器变比、移相角 |
| `BR_STATUS` | 是否投运 |

对于普通线路：

$$
y_{ik}=\frac{1}{r_{ik}+jx_{ik}}.
$$

如果线路总充电电纳为 `b`，通常两端各分一半：

$$
Y_{ff}=y+j\frac{b}{2},\qquad
Y_{tt}=y+j\frac{b}{2},\qquad
Y_{ft}=Y_{tf}=-y.
$$

---

## 3. 标幺值与注入功率

取：

$$
S_{\mathrm{base}}=100\ \mathrm{MVA}.
$$

母线 9 的负荷：

$$
S_{d,9}=\frac{125+j50}{100}=1.25+j0.50\ \mathrm{p.u.}
$$

PYPOWER 的计划母线注入定义为：

$$
S_{\mathrm{bus}}=S_g-S_d.
$$

母线 9 没有在线发电机，因此：

$$
S_{\mathrm{bus},9}=-(1.25+j0.50)\ \mathrm{p.u.}
$$

母线 2、3 是 PV 母线，其有功注入分别约为：

$$
S_{\mathrm{bus},2}=1.63+j0,\qquad
S_{\mathrm{bus},3}=0.85+j0.
$$

无功最终由发电机的调压能力决定，不放在初始指定注入中。

---

## 4. 潮流的未知数和方程

每个母线有：

- 电压幅值 `Vm`；
- 电压相角 `θ`。

9 个母线理论上有 18 个电压自由度。固定条件为：

| 母线 | 固定 |
|---|---|
| 平衡母线 1 | `Vm,θ` |
| PV 母线 2、3 | `Vm` |
| PQ 母线 4-9 | 无电压固定 |

所以待求状态为：

$$
x=
\begin{bmatrix}
\theta_2\\
\theta_3\\
\theta_4\\
\theta_5\\
\theta_6\\
\theta_7\\
\theta_8\\
\theta_9\\
V_{m,4}\\
V_{m,5}\\
V_{m,6}\\
V_{m,7}\\
V_{m,8}\\
V_{m,9}
\end{bmatrix}
\in\mathbb{R}^{14}.
$$

残差同样有 14 维：

$$
F(x)=
\begin{bmatrix}
P_{\mathrm{calc},2}-P_{\mathrm{spec},2}\\
\vdots\\
P_{\mathrm{calc},9}-P_{\mathrm{spec},9}\\
Q_{\mathrm{calc},4}-Q_{\mathrm{spec},4}\\
\vdots\\
Q_{\mathrm{calc},9}-Q_{\mathrm{spec},9}
\end{bmatrix}.
$$

因此每一步求解的是：

$$
J(x_k)\Delta x_k=-F(x_k).
$$

---

## 5. 当前 Python 求解结果

自研求解器 4 次迭代收敛。最终母线电压为：

| 母线 | `Vm` p.u. | `Va` 度 | 类型 |
|---:|---:|---:|---|
| 1 | 1.000000 | 0.000000 | 平衡 |
| 2 | 1.000000 | 9.668741 | PV |
| 3 | 1.000000 | 4.771073 | PV |
| 4 | 0.987007 | -2.406644 | PQ |
| 5 | 0.975472 | -4.017264 | PQ |
| 6 | 1.003375 | 1.925602 | PQ |
| 7 | 0.985645 | 0.621545 | PQ |
| 8 | 0.996185 | 3.799120 | PQ |
| 9 | 0.957621 | -4.349934 | PQ |

发电机结果：

| 发电机 | 接入母线 | `P` MW | `Q` Mvar |
|---:|---:|---:|---:|
| 1 | 1 | 71.954702 | 24.068958 |
| 2 | 2 | 163.000000 | 14.460120 |
| 3 | 3 | 85.000000 | -3.649026 |

功率平衡：

$$
P_{\mathrm{gen}}=319.954702\ \mathrm{MW},
$$

$$
P_{\mathrm{load}}=315.000000\ \mathrm{MW},
$$

$$
P_{\mathrm{loss}}=4.954702\ \mathrm{MW}.
$$

---

## 6. Pandapower 建模

为了排除库自带算例编号差异，项目先把同一个 PYPOWER 字典转成 Pandapower：

```python
from pandapower.converter.pypower.from_ppc import from_ppc
import pandapower as pp

net = from_ppc(ppc, f_hz=60)
pp.runpp(
    net,
    algorithm="nr",
    calculate_voltage_angles=True,
    tolerance_mva=1e-10,
    numba=False,
)
```

Pandapower 表的对应关系：

| PYPOWER | Pandapower |
|---|---|
| `bus` | `net.bus` |
| `gen` 中的 PV 发电机 | `net.gen` |
| 平衡发电机 | `net.ext_grid` |
| `branch` | `net.line` / `net.trafo` |
| 母线潮流结果 | `net.res_bus` |
| 线路潮流结果 | `net.res_line` |

运行：

```bash
.venv/bin/python src/examples/step2_pandapower.py
```

Pandapower 得到相同的 4 次牛顿迭代及相同电压解。

---

## 7. 最容易混淆的三个点

1. **PYPOWER 内部索引是 0 基**。算法中 `bus 0` 对应外部母线 1。
2. **`Sbus` 是发电减负荷**。负荷母线的有功、无功因此为负。
3. **平衡节点的 `P,Q` 和 PV 节点的 `Q` 是结果**，不是预先指定量。
