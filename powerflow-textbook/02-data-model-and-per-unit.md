# 第 2 章 Python 数据模型、标幺值与母线类型

## 2.1 为什么用 PYPOWER 矩阵

PYPOWER 使用二维 NumPy 数组描述母线、发电机和支路。优点是：

- 与电力系统教材中的表格直接对应；
- 列常量使代码可读；
- 易于转换成 Pandapower 表；
- 适合教学时逐行检查数据。

需要牢记：

> PYPOWER 进入内部计算后，母线编号是 0 基连续索引。

外部母线 `1..9` 在内部对应 `0..8`。

## 2.2 母线表

```python
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VM, VA
from pypower.api import case9

ppc = case9()
bus = ppc["bus"]
```

常用列：

| 列常量 | 含义 | 单位 |
|---|---|---|
| `BUS_I` | 母线编号 | - |
| `BUS_TYPE` | 母线类型 | 1/2/3 |
| `PD` | 有功负荷 | MW |
| `QD` | 无功负荷 | Mvar |
| `GS` | 并联电导 | MW |
| `BS` | 并联电纳 | Mvar |
| `VM` | 电压幅值初值 | p.u. |
| `VA` | 电压相角初值 | 度 |
| `BASE_KV` | 基准电压 | kV |

case9 的关键母线：

| 母线 | 类型 | Pd | Qd | Vm0 | Va0 |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0 | 0 | 1.0 | 0 |
| 2 | 2 | 0 | 0 | 1.0 | 0 |
| 3 | 2 | 0 | 0 | 1.0 | 0 |
| 4 | 1 | 0 | 0 | 1.0 | 0 |
| 5 | 1 | 90 | 30 | 1.0 | 0 |
| 6 | 1 | 0 | 0 | 1.0 | 0 |
| 7 | 1 | 100 | 35 | 1.0 | 0 |
| 8 | 1 | 0 | 0 | 1.0 | 0 |
| 9 | 1 | 125 | 50 | 1.0 | 0 |

## 2.3 发电机表

```python
from pypower.idx_gen import GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS
```

| 列常量 | 含义 | 单位 |
|---|---|---|
| `GEN_BUS` | 接入母线 | - |
| `PG` | 有功出力初值 | MW |
| `QG` | 无功出力初值 | Mvar |
| `QMAX,QMIN` | 无功上下限 | Mvar |
| `VG` | 电压幅值设定 | p.u. |
| `GEN_STATUS` | 运行状态 | 1/0 |

case9：

| 发电机 | 母线 | PG0 | QG0 | VG |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0 | 1.0 |
| 2 | 2 | 163 | 0 | 1.0 |
| 3 | 3 | 85 | 0 | 1.0 |

平衡发电机的 PG 初值不会被当作约束。求解结束后，它自动调整到
`71.954702 MW`。

## 2.4 支路表

```python
from pypower.idx_brch import (
    F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, TAP, SHIFT, BR_STATUS
)
```

| 列常量 | 含义 | 单位 |
|---|---|---|
| `F_BUS,T_BUS` | 首端、末端 | - |
| `BR_R,BR_X` | 串联电阻、电抗 | p.u. |
| `BR_B` | 总充电电纳 | p.u. |
| `RATE_A` | 长期容量 | MVA |
| `TAP` | 变比 | p.u. |
| `SHIFT` | 移相角 | 度 |
| `BR_STATUS` | 投运状态 | 1/0 |

case9 没有变压器，所有支路 `TAP=0` 在 PYPOWER 中表示“无变压器变比”。

## 2.5 标幺制

电力系统设备电压等级和容量差异很大，为避免数值尺度失衡，统一使用标幺值：

$$
V_{\mathrm{pu}}=\frac{V}{V_{\mathrm{base}}},
$$

$$
S_{\mathrm{pu}}=\frac{S}{S_{\mathrm{base}}},
$$

$$
Z_{\mathrm{pu}}=\frac{Z}{Z_{\mathrm{base}}}.
$$

本算例：

$$
S_{\mathrm{base}}=100\ \mathrm{MVA}.
$$

母线 9：

$$
P_{d,9}=\frac{125}{100}=1.25\ \mathrm{p.u.},
$$

$$
Q_{d,9}=\frac{50}{100}=0.50\ \mathrm{p.u.}
$$

## 2.6 计划功率注入

PYPOWER 的 `makeSbus` 定义为：

$$
S_{\mathrm{bus},i}=S_{g,i}-S_{d,i}.
$$

### 负荷母线 5

$$
S_{\mathrm{bus},5}=-(0.90+j0.30).
$$

### 负荷母线 7

$$
S_{\mathrm{bus},7}=-(1.00+j0.35).
$$

### 负荷母线 9

$$
S_{\mathrm{bus},9}=-(1.25+j0.50).
$$

### PV 母线 2、3

指定有功：

$$
S_{\mathrm{bus},2}=1.63+j0,
\qquad
S_{\mathrm{bus},3}=0.85+j0.
$$

这里的无功暂时记为 0，但 PV 母线的 `Q` 不是约束，最终会由求解结果回算。

完整 `Sbus`：

| 母线 | Re(Sbus) | Im(Sbus) |
|---:|---:|---:|
| 1 | 0.000000 | 0.000000 |
| 2 | 1.630000 | 0.000000 |
| 3 | 0.850000 | 0.000000 |
| 4 | 0.000000 | 0.000000 |
| 5 | -0.900000 | -0.300000 |
| 6 | 0.000000 | 0.000000 |
| 7 | -1.000000 | -0.350000 |
| 8 | 0.000000 | 0.000000 |
| 9 | -1.250000 | -0.500000 |

## 2.7 母线类型与指定量

| 类型 | P | Q | Vm | θ |
|---|---|---|---|---|
| PQ | 指定 | 指定 | 未知 | 未知 |
| PV | 指定 | 未知 | 指定 | 未知 |
| 平衡 | 未知 | 未知 | 指定 | 指定 |

这就是为什么残差不能简单地对所有母线使用 P、Q 两式。

## 2.8 初始复电压

```python
import numpy as np
from pypower.idx_bus import VM, VA

V0 = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))
```

数学形式：

$$
V_i=V_{m,i}e^{j\theta_i}.
$$

case9 平启动：

$$
V_i=1\angle0^\circ,\qquad i=1,\ldots,9.
$$

## 2.9 Pandapower 数据表

转换后：

```python
net = from_ppc(ppc, f_hz=60)
```

| 原数据 | Pandapower |
|---|---|
| 母线 | `net.bus` |
| 负荷 | `net.load` |
| PV 发电机 | `net.gen` |
| 平衡发电机 | `net.ext_grid` |
| 线路 | `net.line` |

## 2.10 实验

```bash
.venv/bin/python powerflow-textbook/labs/lab02_inspect_case9.py
```

检查：

1. 原始外部编号是否为 `1..9`；
2. 内部 `ref/pv/pq` 是否使用 0 基索引；
3. `Sbus` 的正负是否符合“发电减负荷”；
4. 三个负荷是否合计 `315+j115 Mvar`。
