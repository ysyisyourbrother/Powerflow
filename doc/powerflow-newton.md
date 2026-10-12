# Python 牛顿法潮流求解器剖析

本篇逐段解释 `src/powerflow/newton_raphson.py`。目标是让每个 Python 变量都能在
数学公式中找到位置。

---

## 1. 数据准备

入口：

```python
from powerflow import prepare_case9

problem = prepare_case9()
```

关键结果：

```text
problem.Ybus   # 9×9 复数稀疏矩阵
problem.Sbus   # 9 维计划复功率注入
problem.pv     # array([1, 2])，0 基内部索引
problem.pq     # array([3, 4, 5, 6, 7, 8])
problem.V0     # 9 维初始复电压
```

`Ybus` 由 PYPOWER 的 `makeYbus` 构造：

$$
I_{\mathrm{bus}}=Y_{\mathrm{bus}}V.
$$

`Sbus` 由以下关系构造：

$$
S_{\mathrm{bus}}=\frac{S_g-S_d}{S_{\mathrm{base}}}.
$$

---

## 2. 功率方程

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

数学公式：

$$
P_i=V_i\sum_{k=1}^{n_b}V_k
\left(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}\right),
$$

$$
Q_i=V_i\sum_{k=1}^{n_b}V_k
\left(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}\right),
$$

$$
\theta_{ik}=\theta_i-\theta_k.
$$

对应关系：

| Python | 数学 |
|---|---|
| `Vm` | `V_m` |
| `Va` | `θ` |
| `theta` | `θ_i - θ_k` |
| `G`, `B` | `Ybus` 实部、虚部 |
| `P`, `Q` | 所有母线的计算有功、无功 |

---

## 3. 残差向量

代码：

```python
def mismatch_vector(Ybus, Sbus, V, pv, pq):
    S_calc = V * np.conj(Ybus.dot(V))
    delta_S = S_calc - Sbus
    return np.r_[delta_S[pv].real, delta_S[pq].real, delta_S[pq].imag]
```

先计算：

$$
S_{\mathrm{calc}}(V)=V\circ\operatorname{conj}(Y_{\mathrm{bus}}V).
$$

再计算：

$$
\Delta S=S_{\mathrm{calc}}-S_{\mathrm{spec}}.
$$

残差排列为：

$$
F=
\begin{bmatrix}
\Delta P_{PV}\\
\Delta P_{PQ}\\
\Delta Q_{PQ}
\end{bmatrix}.
$$

当前 case9 初始残差为：

| 行 | 标签 | 数值 |
|---:|---|---:|
| 1 | `ΔP2` | -1.630000000000 |
| 2 | `ΔP3` | -0.850000000000 |
| 3 | `ΔP4` | -2.220446049250e-16 |
| 4 | `ΔP5` | 0.900000000000 |
| 5 | `ΔP6` | 0.000000000000 |
| 6 | `ΔP7` | 1.000000000000 |
| 7 | `ΔP8` | -2.220446049250e-16 |
| 8 | `ΔP9` | 1.250000000000 |
| 9 | `ΔQ4` | -0.167000000000 |
| 10 | `ΔQ5` | 0.042000000000 |
| 11 | `ΔQ6` | -0.283500000000 |
| 12 | `ΔQ7` | 0.171000000000 |
| 13 | `ΔQ8` | -0.227500000000 |
| 14 | `ΔQ9` | 0.259000000000 |

无穷范数：

$$
\|F_0\|_\infty=1.630000.
$$

---

## 4. 雅可比四个分块

代码顺序：

```text
J11 = ∂P/∂θ
J12 = ∂P/∂Vm
J21 = ∂Q/∂θ
J22 = ∂Q/∂Vm
```

### 4.1 角度-角度块

对角：

$$
\frac{\partial P_i}{\partial\theta_i}
=-Q_i-B_{ii}V_{m,i}^2,
$$

$$
\frac{\partial Q_i}{\partial\theta_i}
=P_i-G_{ii}V_{m,i}^2.
$$

非对角：

$$
\frac{\partial P_i}{\partial\theta_k}
=V_{m,i}V_{m,k}
\left(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}\right),
$$

$$
\frac{\partial Q_i}{\partial\theta_k}
=-V_{m,i}V_{m,k}
\left(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}\right).
$$

Python：

```python
J[row, col] = -Q[i] - B[i, i] * Vm[i] ** 2

J[row, col] = Vm[i] * Vm[k] * (
    G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
)
```

### 4.2 角度-幅值块

对角：

$$
\frac{\partial P_i}{\partial V_{m,i}}
=\frac{P_i}{V_{m,i}}+G_{ii}V_{m,i},
$$

$$
\frac{\partial Q_i}{\partial V_{m,i}}
=\frac{Q_i}{V_{m,i}}-B_{ii}V_{m,i}.
$$

非对角：

$$
\frac{\partial P_i}{\partial V_{m,k}}
=V_{m,i}
\left(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}\right),
$$

$$
\frac{\partial Q_i}{\partial V_{m,k}}
=V_{m,i}
\left(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}\right).
$$

代码完全按照上述四式分支计算。

---

## 5. 迭代主循环

核心代码：

```python
mismatch = mismatch_vector(problem.Ybus, problem.Sbus, V, pv, pq)

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
```

数学顺序：

1. 计算 `F(x_k)`；
2. 计算 `J(x_k)`；
3. 解 `J Δx = -F`；
4. 更新角度和幅值；
5. 由 `V=Vm·exp(jθ)` 重建复电压；
6. 重新计算残差；
7. 检查 `||F||∞ < tol`。

向量分段：

$$
\Delta x=
\begin{bmatrix}
\Delta\theta_{PV}\\
\Delta\theta_{PQ}\\
\Delta V_{m,PQ}
\end{bmatrix}.
$$

Python 切片：

```text
correction[:npv]                   -> Δθ_PV
correction[npv:npv + npq]          -> Δθ_PQ
correction[npv + npq:]              -> ΔVm_PQ
```

---

## 6. case9 实际迭代

| 迭代 | `||F||∞`（更新前） | `||dx||∞` | `||F||∞`（更新后） |
|---:|---:|---:|---:|
| 0 | 1.63000000e+00 | - | - |
| 1 | 1.63000000e+00 | 1.71821801e-01 | 1.67050706e-01 |
| 2 | 1.67050706e-01 | 1.36397876e-02 | 1.88742735e-03 |
| 3 | 1.88742735e-03 | 2.05817880e-04 | 5.71435643e-07 |
| 4 | 5.71435643e-07 | 5.33351509e-08 | 5.48450174e-14 |

从第 4 次更新后的残差可以看出：

$$
5.4845\times10^{-14}<10^{-10},
$$

因此收敛。

---

## 7. 结果回算

牛顿法只返回最终复电压。回算代码：

```python
external = build_external_result(problem, result)
```

内部执行：

```python
bus, gen, branch = pfsoln(
    base_mva,
    bus,
    gen,
    branch,
    Ybus,
    Yf,
    Yt,
    result.voltage,
    ref,
    pv,
    pq,
)
```

三个核心回算公式：

母线复功率：

$$
S_{\mathrm{bus},i}=V_i\operatorname{conj}\left(\sum_kY_{ik}V_k\right).
$$

发电机无功：

$$
Q_g=\operatorname{Im}(S_{\mathrm{bus}})S_{\mathrm{base}}+Q_d.
$$

平衡发电机有功：

$$
P_g=\operatorname{Re}(S_{\mathrm{bus}})S_{\mathrm{base}}+P_d.
$$

最终结果：

| 发电机 | 母线 | P MW | Q Mvar |
|---:|---:|---:|---:|
| 1 | 1 | 71.954702 | 24.068958 |
| 2 | 2 | 163.000000 | 14.460120 |
| 3 | 3 | 85.000000 | -3.649026 |

---

## 8. 与 PYPOWER 的雅可比互验

PYPOWER 使用复数矩阵技巧：

```python
dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
```

再取实部和虚部拼成实数雅可比。项目测试：

```python
J_our = jacobian_polar(Ybus, V0, pv, pq)
J_pypower = assemble_jacobian_from_dSbus_dV(...)
assert np.max(np.abs(J_our - J_pypower)) < 1e-12
```

复杂度数矩阵推导和逐项公式在 case9 上得到相同矩阵，最大差为 `0.0`。

---

## 9. 学习时重点观察

1. `F` 只包含 PV/PQ 有功和 PQ 无功；
2. `J` 的行列顺序必须与 `F`、`dx` 完全一致；
3. 每次迭代后重新计算 `F`，而不是沿旧残差继续；
4. 平衡节点不进入状态更新，但参与 `Ybus·V`；
5. PV 母线只更新相角，不更新幅值；
6. PQ 母线同时更新相角和幅值。
