# 第 6 章 Python 代码与公式逐步对照

本篇以 `src/powerflow/newton_raphson.py` 为主，按实际执行顺序解释每一行代码。读完应能
独立复写整个求解器。

## 6.1 调用入口

```python
from powerflow import prepare_case9, solve_newton_raphson, build_external_result

problem = prepare_case9()
result = solve_newton_raphson(problem, tol=1e-10, max_iter=20)
external = build_external_result(problem, result)
```

对应阶段：

| 调用 | 作用 |
|---|---|
| `prepare_case9` | 读取数据、构造 `Ybus/Sbus/V0`、划分母线 |
| `solve_newton_raphson` | 完整牛顿迭代 |
| `build_external_result` | 回算发电机和线路，并恢复外部编号 |

## 6.2 稠密化辅助函数

```python
def _dense(matrix):
    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=complex)
    return np.asarray(matrix, dtype=complex)
```

`Ybus` 是 SciPy 稀疏矩阵。教学时转到稠密数组方便打印，但对大赛规模问题应保留稀疏结构。

## 6.3 P、Q 方程

代码：

```python
def polar_power_equations(Ybus, V):
    Y = _dense(Ybus)
    G = Y.real
    B = Y.imag
    Vm = np.abs(V)
    Va = np.angle(V)
    theta = Va[:, None] - Va[None, :]

    P = Vm * np.sum(Vm[None, :] * (G * np.cos(theta) + B * np.sin(theta)), axis=1)
    Q = Vm * np.sum(Vm[None, :] * (G * np.sin(theta) - B * np.cos(theta)), axis=1)
    return P, Q
```

公式：

$$
P_i=V_i\sum_kV_k(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}),
$$

$$
Q_i=V_i\sum_kV_k(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}).
$$

逐项对照：

| 代码 | 公式 | 说明 |
|---|---|---|
| `G = Y.real` | `G_ik` | 导纳实部 |
| `B = Y.imag` | `B_ik` | 导纳虚部 |
| `Vm = np.abs(V)` | `V_i,V_k` | 电压幅值 |
| `Va = np.angle(V)` | `θ_i,θ_k` | 电压相角 |
| `theta[i,k]` | `θ_i-θ_k` | 相角差 |
| `np.cos(theta)` | `cos θ_ik` | 余弦项 |
| `np.sin(theta)` | `sin θ_ik` | 正弦项 |
| `axis=1` | `Σ_k` | 沿母线维求和 |

### 为什么用广播

```python
Vm[None, :]
```

把行向量复制成 `nb × nb`：

$$
V_k\rightarrow V_{ik}^{\text{row}}.
$$

`theta` 本身就是：

```text
theta[i, j] = Va[i] - Va[j]
```

因此无需双重 Python 循环。

## 6.4 残差函数

代码：

```python
def mismatch_vector(Ybus, Sbus, V, pv, pq):
    V = np.asarray(V, dtype=complex)
    S_calc = V * np.conj(Ybus.dot(V))
    delta_S = S_calc - Sbus
    return np.r_[delta_S[pv].real, delta_S[pq].real, delta_S[pq].imag]
```

公式：

$$
S_{\mathrm{calc}}
=V\circ\operatorname{conj}(Y_{\mathrm{bus}}V),
$$

$$
\Delta S=S_{\mathrm{calc}}-S_{\mathrm{spec}}.
$$

残差：

$$
F=
\begin{bmatrix}
\Delta P_{PV}\\
\Delta P_{PQ}\\
\Delta Q_{PQ}
\end{bmatrix}.
$$

对应：

| 代码 | 公式 |
|---|---|
| `Ybus.dot(V)` | `Ybus V` |
| `np.conj(...)` | `conj(...)` |
| `V * ...` | `diag(V) ...` |
| `delta_S[pv].real` | `ΔP_PV` |
| `delta_S[pq].real` | `ΔP_PQ` |
| `delta_S[pq].imag` | `ΔQ_PQ` |

## 6.5 雅可比函数总览

```python
def jacobian_polar(Ybus, V, pv, pq):
    Y = _dense(Ybus)
    G = Y.real
    B = Y.imag
    Vm = np.abs(V)
    Va = np.angle(V)
    P, Q = polar_power_equations(Ybus, V)

    pvpq = np.r_[pv, pq]
    n_angle = len(pvpq)
    n_pq = len(pq)
    J = np.zeros((n_angle + n_pq, n_angle + n_pq), dtype=float)
```

行列顺序：

$$
[\theta_{PV},\theta_{PQ},V_{m,PQ}]
$$

因此：

```text
J11：行 [PV,PQ]，列 [PV,PQ]
J12：行 [PV,PQ]，列 [PQ]
J21：行 [PQ]，列 [PV,PQ]
J22：行 [PQ]，列 [PQ]
```

## 6.6 `J11 = ∂P/∂θ`

```python
for row, i in enumerate(pvpq):
    for col, k in enumerate(pvpq):
        if i == k:
            J[row, col] = -Q[i] - B[i, i] * Vm[i] ** 2
        else:
            theta_ik = Va[i] - Va[k]
            J[row, col] = Vm[i] * Vm[k] * (
                G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
            )
```

公式：

$$
\frac{\partial P_i}{\partial\theta_i}
=-Q_i-B_{ii}V_{m,i}^2,
$$

$$
\frac{\partial P_i}{\partial\theta_k}
=V_{m,i}V_{m,k}
(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}).
$$

## 6.7 `J12 = ∂P/∂Vm`

```python
for row, i in enumerate(pvpq):
    for col, k in enumerate(pq):
        if i == k:
            J[row, n_angle + col] = P[i] / Vm[i] + G[i, i] * Vm[i]
        else:
            theta_ik = Va[i] - Va[k]
            J[row, n_angle + col] = Vm[i] * (
                G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik)
            )
```

公式：

$$
\frac{\partial P_i}{\partial V_{m,i}}
=\frac{P_i}{V_{m,i}}+G_{ii}V_{m,i},
$$

$$
\frac{\partial P_i}{\partial V_{m,k}}
=V_{m,i}(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}).
$$

## 6.8 `J21 = ∂Q/∂θ`

```python
for row, i in enumerate(pq):
    for col, k in enumerate(pvpq):
        if i == k:
            J[n_angle + row, col] = P[i] - G[i, i] * Vm[i] ** 2
        else:
            theta_ik = Va[i] - Va[k]
            J[n_angle + row, col] = -Vm[i] * Vm[k] * (
                G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik)
            )
```

公式：

$$
\frac{\partial Q_i}{\partial\theta_i}
=P_i-G_{ii}V_{m,i}^2,
$$

$$
\frac{\partial Q_i}{\partial\theta_k}
=-V_{m,i}V_{m,k}
(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}).
$$

## 6.9 `J22 = ∂Q/∂Vm`

```python
for row, i in enumerate(pq):
    for col, k in enumerate(pq):
        if i == k:
            J[n_angle + row, n_angle + col] = Q[i] / Vm[i] - B[i, i] * Vm[i]
        else:
            theta_ik = Va[i] - Va[k]
            J[n_angle + row, n_angle + col] = Vm[i] * (
                G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
            )
```

公式：

$$
\frac{\partial Q_i}{\partial V_{m,i}}
=\frac{Q_i}{V_{m,i}}-B_{ii}V_{m,i},
$$

$$
\frac{\partial Q_i}{\partial V_{m,k}}
=V_{m,i}(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}).
$$

## 6.10 牛顿迭代完整代码

```python
def solve_newton_raphson(problem, tol=1e-10, max_iter=20):
    V = problem.V0.copy()
    pv = problem.pv
    pq = problem.pq
    npv = len(pv)
    npq = len(pq)

    mismatch = mismatch_vector(problem.Ybus, problem.Sbus, V, pv, pq)
    norm = np.linalg.norm(mismatch, ord=np.inf)

    if norm < tol:
        return NewtonResult(...)

    for iteration in range(1, max_iter + 1):
        P, Q = polar_power_equations(problem.Ybus, V)
        J = jacobian_polar(problem.Ybus, V, pv, pq)

        correction = np.linalg.solve(J, -mismatch)

        Vm = np.abs(V)
        Va = np.angle(V)
        Va[pv] += correction[:npv]
        Va[pq] += correction[npv:npv + npq]
        Vm[pq] += correction[npv + npq:]
        V = Vm * np.exp(1j * Va)

        mismatch = mismatch_vector(problem.Ybus, problem.Sbus, V, pv, pq)
        norm = np.linalg.norm(mismatch, ord=np.inf)

        if norm < tol:
            break
```

## 6.11 状态更新逐行对照

### PV 相角

```python
Va[pv] += correction[:npv]
```

公式：

$$
\theta_{PV}^{(k+1)}
=\theta_{PV}^{(k)}
+\Delta\theta_{PV}^{(k)}.
$$

PV 电压幅值不变。

### PQ 相角

```python
Va[pq] += correction[npv:npv + npq]
```

公式：

$$
\theta_{PQ}^{(k+1)}
=\theta_{PQ}^{(k)}
+\Delta\theta_{PQ}^{(k)}.
$$

### PQ 幅值

```python
Vm[pq] += correction[npv + npq:]
```

公式：

$$
V_{m,PQ}^{(k+1)}
=V_{m,PQ}^{(k)}
+\Delta V_{m,PQ}^{(k)}.
$$

### 重建复电压

```python
V = Vm * np.exp(1j * Va)
```

公式：

$$
V_i=V_{m,i}e^{j\theta_i}.
$$

## 6.12 数据准备代码

```python
ppc = case9()
ppci = ext2int(ppc)
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
Sbus = makeSbus(baseMVA, bus, gen)
ref, pv, pq = bustypes(bus, gen)
V0 = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))
```

对应：

| 代码 | 数学/工程含义 |
|---|---|
| `ext2int` | 外部编号转连续内部索引 |
| `makeYbus` | 构造 `Ybus` |
| `makeSbus` | 构造 `Sg-Sd` |
| `bustypes` | 找 PQ/PV/平衡母线 |
| `V0` | 构造 `Vm exp(jθ)` |

## 6.13 回算代码

```python
bus, gen, branch = pfsoln(
    base_mva, bus, gen, branch,
    Ybus, Yf, Yt, V, ref, pv, pq
)
```

数学关系：

母线注入：

$$
S_{\mathrm{bus},i}
=V_i\operatorname{conj}\left(\sum_kY_{ik}V_k\right).
$$

发电机无功：

$$
Q_g=\operatorname{Im}(S_{\mathrm{bus}})S_{\mathrm{base}}+Q_d.
$$

平衡发电机有功：

$$
P_g=\operatorname{Re}(S_{\mathrm{bus}})S_{\mathrm{base}}+P_d.
$$

线路首端潮流：

$$
S_f=V_f\operatorname{conj}(I_f)S_{\mathrm{base}}.
$$

线路末端潮流：

$$
S_t=V_t\operatorname{conj}(I_t)S_{\mathrm{base}}.
$$

## 6.14 代码与公式一一对应的检查表

| 数学步骤 | Python |
|---|---|
| `I=YV` | `Ybus.dot(V)` |
| `S=V.*conj(I)` | `V * np.conj(Ybus.dot(V))` |
| `ΔS=S_calc-S_spec` | `delta_S = S_calc - Sbus` |
| `F=[ΔP,ΔQ]` | `np.r_[...]` |
| `J=∂F/∂x` | `jacobian_polar` |
| `JΔx=-F` | `np.linalg.solve(J, -mismatch)` |
| `x=x+Δx` | 三个切片更新 |
| `V=Vm exp(jθ)` | `Vm * np.exp(1j * Va)` |
| `||F||∞` | `np.linalg.norm(F, np.inf)` |

## 6.15 实验

```bash
.venv/bin/python powerflow-textbook/labs/lab04_first_iteration.py
.venv/bin/python powerflow-textbook/labs/lab05_trace_newton.py
.venv/bin/python powerflow-textbook/labs/lab06_finite_difference.py
```

检查每个输出数组的行列顺序，并与本章对照。
