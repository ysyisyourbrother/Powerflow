# 附录 A：Python / PYPOWER / Pandapower 速查

## A.1 安装与自检

```bash
./install.sh
.venv/bin/python src/examples/smoke_test.py
```

手动：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## A.2 PYPOWER 常用导入

```python
from pypower.api import (
    case9, case14, case30, case57,
    ext2int, int2ext, bustypes,
    makeYbus, makeSbus, pfsoln,
    runpf, ppoption,
    newtonpf, fdpf, gausspf, makeB,
)
```

## A.3 常量

母线：

```python
from pypower.idx_bus import (
    BUS_I, BUS_TYPE, PD, QD, GS, BS, VM, VA, BASE_KV
)
```

发电机：

```python
from pypower.idx_gen import (
    GEN_BUS, PG, QG, QMAX, QMIN, VG, GEN_STATUS, PMAX, PMIN
)
```

支路：

```python
from pypower.idx_brch import (
    F_BUS, T_BUS, BR_R, BR_X, BR_B,
    RATE_A, TAP, SHIFT, BR_STATUS,
    PF, QF, PT, QT
)
```

## A.4 读取 case9

```python
from pypower.api import case9

ppc = case9()
print(ppc["baseMVA"])
print(ppc["bus"].shape)
print(ppc["gen"].shape)
print(ppc["branch"].shape)
```

## A.5 内部编号

```python
from pypower.api import ext2int, int2ext

ppci = ext2int(ppc)
ppc_external = int2ext(ppci)
```

内部索引通常为 `0..nb-1`。算法变量使用内部索引，结果展示时恢复外部编号。

## A.6 母线分类

```python
from pypower.api import bustypes

ref, pv, pq = bustypes(ppci["bus"], ppci["gen"])
```

case9：

```text
ref = [0]
pv  = [1, 2]
pq  = [3, 4, 5, 6, 7, 8]
```

## A.7 构造矩阵和注入

```python
from pypower.api import makeYbus, makeSbus

Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
Sbus = makeSbus(ppci["baseMVA"], ppci["bus"], ppci["gen"])
```

## A.8 初始复电压

```python
import numpy as np
from pypower.idx_bus import VM, VA

V0 = ppci["bus"][:, VM] * np.exp(1j * np.deg2rad(ppci["bus"][:, VA]))
```

## A.9 运行 PYPOWER

```python
from pypower.api import runpf, ppoption

options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-10)
result, success = runpf(ppc, options)
```

算法编号：

| `PF_ALG` | 算法 |
|---:|---|
| 1 | Newton |
| 2 | Fast-decoupled XB |
| 3 | Fast-decoupled BX |
| 4 | Gauss-Seidel |

## A.10 项目中运行自研求解器

```python
from powerflow import prepare_case9, solve_newton_raphson, build_external_result

problem = prepare_case9()
result = solve_newton_raphson(problem, tol=1e-10, max_iter=20)
external = build_external_result(problem, result)
```

## A.11 读取迭代轨迹

```python
print(result.initial_mismatch)
for step in result.steps:
    print(step.iteration)
    print(step.norm_before)
    print(step.jacobian)
    print(step.correction)
    print(step.mismatch_after)
```

## A.12 Pandapower 常用导入

```python
import pandapower as pp
import pandapower.networks as pn
from pandapower.converter.pypower.from_ppc import from_ppc
```

## A.13 PYPOWER 转 Pandapower

```python
net = from_ppc(ppc, f_hz=60)
pp.runpp(
    net,
    algorithm="nr",
    calculate_voltage_angles=True,
    tolerance_mva=1e-10,
    numba=False,
)
```

## A.14 Pandapower 结果表

```python
print(net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]])
print(net.res_gen[["p_mw", "q_mvar"]])
print(net.res_ext_grid[["p_mw", "q_mvar"]])
print(net.res_line[["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar"]])
```

## A.15 创建 Pandapower 新网络

```python
net = pp.create_empty_network(sn_mva=100)
b1 = pp.create_bus(net, vn_kv=110)
b2 = pp.create_bus(net, vn_kv=110)
pp.create_ext_grid(net, b1, vm_pu=1.0, va_degree=0.0)
pp.create_load(net, b2, p_mw=50, q_mvar=20)
pp.create_line_from_parameters(
    net, b1, b2, length_km=20,
    r_ohm_per_km=0.05, x_ohm_per_km=0.20,
    c_nf_per_km=10, max_i_ka=0.5,
)
pp.runpp(net, calculate_voltage_angles=True)
```

## A.16 NumPy 常用操作

```python
import numpy as np

V = np.array([1 + 0j, 0.98 - 0.02j])
Vm = np.abs(V)
Va = np.angle(V)
V_restored = Vm * np.exp(1j * Va)

S = V * np.conj(Ybus.dot(V))
norm_inf = np.linalg.norm(F, ord=np.inf)
dx = np.linalg.solve(J, -F)
```

## A.17 常见错误

### 错误 1：把 `Sbus` 当成负荷

正确：

```python
Sbus = generation - load
```

负荷母线通常为负。

### 错误 2：把角度单位和弧度混用

PYPOWER 表格中的 `VA` 通常为度，算法内部为弧度。

### 错误 3：索引错位

状态向量、残差和雅可比必须使用同一个母线顺序：

```python
pvpq = np.r_[pv, pq]
```

### 错误 4：忘记重新计算残差

每次更新 `V` 后必须：

```python
mismatch = mismatch_vector(Ybus, Sbus, V, pv, pq)
```

### 错误 5：回算结果仍用内部编号

最终展示前必须：

```python
external = int2ext(internal)
```
