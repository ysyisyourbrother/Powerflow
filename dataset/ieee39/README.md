# IEEE 39 节点系统原始数据

本目录已经放入 IEEE 39 节点 New England 系统的原始数据、参考潮流结果和导出脚本。

位置：仓库根目录下的 `dataset/ieee39/`。

## 1. 数据规模

| 元素 | 数量 |
|---|---:|
| 母线 | 39 |
| 发电机 | 10 |
| 支路 | 46 |
| 负荷 | 21 |
| 基准容量 | 100 MVA |
| 负荷有功 | 6254.23 MW |
| 负荷无功 | 1387.10 Mvar |

数据表示：

| 表 | 内容 |
|---|---|
| `bus` | 39 个母线 |
| `gen` | 10 台发电机 |
| `branch` | 46 条支路 |

PYPOWER 把线路和变压器统一放在 `branch` 表：46 条支路 = 34 条线路 +
12 条变压器等值支路，靠 `tap_pu` 列区分（`tap_pu = 0` 表示线路）。
这 12 条里有 1 条变比恰好为 1.0（23→36），电气上与线路等价，但类型仍为
变压器支路；按“变比 ≠ 1”数则是 11 台真正改变电压的变压器。

## 2. 数据来源

本数据从本机已安装的 PYPOWER 中直接提取：

- PYPOWER `5.1.21`：`pypower.case39`

PYPOWER 数据文件是已求解的 New England 系统版本，包含以下处理：

- 发电机母线连续编号；
- 为所有发电机补充 `Pmin=0`；
- 为母线 31、39 的发电机补充无功限值；
- 为母线 39 补充发电机、负荷和电压设定；
- 补充线路容量、电压上下限和二次发电成本；
- 将母线 34 的 `Pmax` 从 308 MW 修正为 508 MW；
- 重新求解潮流。

上游代码中记录的历史参考：

- G. W. Bills et al., On-Line Stability Analysis Study, EEI RP90-1, 1970
- M. A. Pai, Energy Function Analysis for Power System Stability, 1989
- Athay, Podmore, Virmani, A Practical Method for the Direct Analysis of Transient Stability, 1979

对应 Python 上游：

- PYPOWER case39：<https://github.com/rwl/PYPOWER/blob/master/pypower/case39.py>

## 3. 目录内容

```text
dataset/ieee39/
├── README.md
├── manifest.json
├── checksums.sha256
├── raw/
│   └── pypower/
│       ├── case39.json
│       ├── pypower_case39_arrays.npz
│       ├── bus.csv
│       ├── gen.csv
│       ├── branch.csv
│       └── gencost.csv
├── reference/
│   ├── summary.json
│   ├── bus_solution_pypower.csv
│   ├── gen_solution_pypower.csv
│   └── branch_solution_pypower.csv
├── processed/
│   └── competition_v1/         # 赛事对齐数据集（当前主用）
└── scripts/
    └── export_ieee39.py
```

赛事对齐数据集的字段、单位、生成命令和自检结果，见
[`processed/competition_v1/README.md`](processed/competition_v1/README.md)；
数据集总入口见 [`../README.md`](../README.md)。

## 4. 原始表

### bus.csv

主要列：

| 列 | 含义 |
|---|---|
| `bus_i` | 母线编号，1-39 |
| `bus_type` | 1=PQ，2=PV，3=平衡 |
| `pd_mw,qd_mvar` | 负荷 |
| `gs_mw,bs_mvar` | 并联元件 |
| `area` | 区域 |
| `vm_pu,va_degree` | 电压初值 |
| `base_kv` | 基准电压 |
| `vmax_pu,vmin_pu` | 电压限值 |

### gen.csv

主要列：

| 列 | 含义 |
|---|---|
| `gen_bus` | 发电机母线 |
| `pg_mw,qg_mvar` | 发电机初始出力 |
| `qmax_mvar,qmin_mvar` | 无功限值 |
| `vg_pu` | 电压设定 |
| `status` | 运行状态 |
| `pmax_mw,pmin_mw` | 有功限值 |

### branch.csv

PYPOWER 把线路和变压器统一放在 branch 表：

| 列 | 含义 |
|---|---|
| `f_bus,t_bus` | 首端、末端母线 |
| `r_pu,x_pu,b_pu` | 串联电阻、电抗、充电电纳 |
| `rate_a_mva,rate_b_mva,rate_c_mva` | 容量 |
| `tap_pu` | 变比 |
| `shift_degree` | 移相角 |
| `angmin_degree,angmax_degree` | 相角差限值 |

`tap_pu != 0` 的行是变压器支路。

### gencost.csv

每个发电机使用三阶多项式成本：

$$
C(P)=c_2P^2+c_1P+c_0.
$$

当前数据：

```text
c2=0.01, c1=0.3, c0=0.2
```

## 5. 参考潮流结果

求解设置：

```text
算法       Newton-Raphson
容差       1e-10
基准容量   100 MVA
```

结果：

| 指标 | 数值 |
|---|---:|
| 收敛 | true |
| 总负荷 | 6254.23 MW |
| 总发电 | 6297.871125760811 MW |
| 网络有功损耗 | 43.64112576081152 MW |
| 最低电压 | 0.982 p.u. |
| 最高电压 | 1.0636 p.u. |
| 最低电压相角 | -14.535256° |
| 最高电压相角 | 4.468437° |

完整摘要：

```bash
cat dataset/ieee39/reference/summary.json
```

## 6. 直接加载

### 从 PYPOWER 求解

```python
from pypower.api import case39, runpf, ppoption

ppc = case39()
result, success = runpf(
    ppc,
    ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-10),
)
```

### 从 CSV 读取原始表

```python
import pandas as pd

bus = pd.read_csv("dataset/ieee39/raw/pypower/bus.csv")
gen = pd.read_csv("dataset/ieee39/raw/pypower/gen.csv")
branch = pd.read_csv("dataset/ieee39/raw/pypower/branch.csv")
```

## 7. 重新生成

```bash
.venv/bin/python dataset/ieee39/scripts/export_ieee39.py
```

脚本会重新导出原始表、参考解并更新：

- `manifest.json`
- `checksums.sha256`

## 8. 已知数据特征

- 母线 31 是平衡母线。
- 母线 30、32-39 是 PV 母线。
- 其余 29 个母线是 PQ 母线。
- 线路和变压器均处于投运状态。
- 母线 9 和 24 的 `Qd` 为负，表示这些负荷点包含无功补偿特性。
- 原始 PYPOWER 文档记录：母线 31 的 `PG=677.871 MW` 超过其 `Pmax=646 MW`，
  母线 37 的 `QG=-1.36945 Mvar` 低于其 `Qmin=0 Mvar`。保留原始数据时不要悄悄修正。
