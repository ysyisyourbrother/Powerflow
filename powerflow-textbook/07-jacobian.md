# 第 7 章 雅可比矩阵：推导、Python 实现与验证

## 7.1 雅可比的数学定义

设：

$$
F(x)=
\begin{bmatrix}
F_1(x_1,\ldots,x_{14})\\
\vdots\\
F_{14}(x_1,\ldots,x_{14})
\end{bmatrix}.
$$

雅可比矩阵：

$$
J_{ij}=\frac{\partial F_i}{\partial x_j}.
$$

状态顺序：

$$
x=
\begin{bmatrix}
\theta_{PV}\\
\theta_{PQ}\\
V_{m,PQ}
\end{bmatrix}.
$$

残差顺序：

$$
F=
\begin{bmatrix}
P_{PV\cup PQ}-P_{\mathrm{spec}}\\
Q_{PQ}-Q_{\mathrm{spec}}
\end{bmatrix}.
$$

所以：

$$
J=
\begin{bmatrix}
\dfrac{\partial P}{\partial\theta} &
\dfrac{\partial P}{\partial V_m}\\[1em]
\dfrac{\partial Q}{\partial\theta} &
\dfrac{\partial Q}{\partial V_m}
\end{bmatrix}
=
\begin{bmatrix}
J_{11} & J_{12}\\
J_{21} & J_{22}
\end{bmatrix}.
$$

## 7.2 分块维度

case9：

| 分块 | 行 | 列 | 维度 |
|---|---|---|---:|
| J11 | PV+PQ | PV+PQ | 8×8 |
| J12 | PV+PQ | PQ | 8×6 |
| J21 | PQ | PV+PQ | 6×8 |
| J22 | PQ | PQ | 6×6 |

合并：

$$
J\in\mathbb{R}^{14\times14}.
$$

## 7.3 极坐标 P、Q 公式

记：

$$
\theta_{ik}=\theta_i-\theta_k,
$$

$$
Y_{ik}=G_{ik}+jB_{ik}.
$$

则：

$$
P_i=V_{m,i}\sum_kV_{m,k}
(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}),
$$

$$
Q_i=V_{m,i}\sum_kV_{m,k}
(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}).
$$

所有偏导都可从这两个式子推出。

## 7.4 `∂P/∂θ`

### 对角

$$
\frac{\partial P_i}{\partial\theta_i}
=-Q_i-B_{ii}V_{m,i}^2.
$$

注意右端直接使用当前计算出来的 `Q_i`，不是固定注入。

### 非对角

$$
\frac{\partial P_i}{\partial\theta_k}
=V_{m,i}V_{m,k}
(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}),
\qquad i\ne k.
$$

## 7.5 `∂Q/∂θ`

### 对角

$$
\frac{\partial Q_i}{\partial\theta_i}
=P_i-G_{ii}V_{m,i}^2.
$$

### 非对角

$$
\frac{\partial Q_i}{\partial\theta_k}
=-V_{m,i}V_{m,k}
(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}),
\qquad i\ne k.
$$

## 7.6 `∂P/∂Vm`

### 对角

$$
\frac{\partial P_i}{\partial V_{m,i}}
=\frac{P_i}{V_{m,i}}+G_{ii}V_{m,i}.
$$

### 非对角

$$
\frac{\partial P_i}{\partial V_{m,k}}
=V_{m,i}
(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}),
\qquad i\ne k.
$$

## 7.7 `∂Q/∂Vm`

### 对角

$$
\frac{\partial Q_i}{\partial V_{m,i}}
=\frac{Q_i}{V_{m,i}}-B_{ii}V_{m,i}.
$$

### 非对角

$$
\frac{\partial Q_i}{\partial V_{m,k}}
=V_{m,i}
(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}),
\qquad i\ne k.
$$

## 7.8 Python 实现映射

### 角度-角度

```python
if i == k:
    J[row, col] = -Q[i] - B[i, i] * Vm[i] ** 2
else:
    theta_ik = Va[i] - Va[k]
    J[row, col] = Vm[i] * Vm[k] * (
        G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
    )
```

### 角度-幅值

```python
if i == k:
    J[row, n_angle + col] = P[i] / Vm[i] + G[i, i] * Vm[i]
else:
    theta_ik = Va[i] - Va[k]
    J[row, n_angle + col] = Vm[i] * (
        G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik)
    )
```

### 无功-角度

```python
if i == k:
    J[n_angle + row, col] = P[i] - G[i, i] * Vm[i] ** 2
else:
    theta_ik = Va[i] - Va[k]
    J[n_angle + row, col] = -Vm[i] * Vm[k] * (
        G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik)
    )
```

### 无功-幅值

```python
if i == k:
    J[n_angle + row, n_angle + col] = Q[i] / Vm[i] - B[i, i] * Vm[i]
else:
    theta_ik = Va[i] - Va[k]
    J[n_angle + row, n_angle + col] = Vm[i] * (
        G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
    )
```

## 7.9 case9 初始雅可比

初始电压全为 `1∠0°`。下面每一行对应：

```text
[ΔP2, ΔP3, ΔP4, ΔP5, ΔP6, ΔP7, ΔP8, ΔP9,
 ΔQ4, ΔQ5, ΔQ6, ΔQ7, ΔQ8, ΔQ9]
```

每一列对应：

```text
[θ2, θ3, θ4, θ5, θ6, θ7, θ8, θ9,
 Vm4, Vm5, Vm6, Vm7, Vm8, Vm9]
```

$$
J_0=
\begin{bmatrix}
16.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 & 0.000000 & -16.000000 & 0.000000 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 17.064846 & 0 & 0 & -17.064846 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 39.475889 & -10.510682 & 0 & 0 & 0 & -11.604096 & 3.307379 & -1.942191 & 0 & 0 & 0 & -1.365188\\
0 & 0 & -10.510682 & 16.098927 & -5.588245 & 0 & 0 & 0 & -1.942191 & 3.224200 & -1.282009 & 0 & 0 & 0\\
0 & -17.064846 & 0 & -5.588245 & 32.437362 & -9.784270 & 0 & 0 & 0 & -1.282009 & 2.437097 & -1.155087 & 0 & 0\\
0 & 0 & 0 & 0 & -9.784270 & 23.482249 & -13.697979 & 0 & 0 & 0 & -1.155087 & 2.772210 & -1.617122 & 0\\
-16.000000 & 0 & 0 & 0 & 0 & -13.697979 & 35.673113 & -5.975135 & 0 & 0 & 0 & -1.617122 & 2.804727 & -1.187604\\
0 & 0 & -11.604096 & 0 & 0 & 0 & -5.975135 & 17.579230 & -1.365188 & 0 & 0 & 0 & -1.187604 & 2.552792\\
0 & 0 & -3.307379 & 1.942191 & 0 & 0 & 0 & 1.365188 & 39.141889 & -10.510682 & 0 & 0 & 0 & -11.604096\\
0 & 0 & 1.942191 & -3.224200 & 1.282009 & 0 & 0 & 0 & -10.510682 & 15.582927 & -5.588245 & 0 & 0 & 0\\
0 & 0 & 0 & 1.282009 & -2.437097 & 1.155087 & 0 & 0 & 0 & -5.588245 & 31.870362 & -9.784270 & 0 & 0\\
0 & 0 & 0 & 0 & 1.155087 & -2.772210 & 1.617122 & 0 & 0 & 0 & -9.784270 & 23.124249 & -13.697979 & 0\\
0 & 0 & 0 & 0 & 0 & 1.617122 & -2.804727 & 1.187604 & 0 & 0 & 0 & -13.697979 & 35.218113 & -5.975135\\
0 & 0 & 1.365188 & 0 & 0 & 0 & 1.187604 & -2.552792 & -11.604096 & 0 & 0 & 0 & -5.975135 & 17.097230
\end{bmatrix}.
$$

## 7.10 如何从矩阵结构看拓扑

`J` 的非零位置对应 P/Q 方程对状态变量的依赖。

例如：

- 母线 2 的 P 方程只依赖 `θ2` 和 `θ8`；
- 母线 9 的 P 方程依赖 `θ4`、`θ8`、`θ9` 及幅值 `Vm4,Vm8,Vm9`；
- 因此 `J` 的稀疏结构与 `Ybus` 一致。

生产求解器不会构造完整稠密矩阵，而是保留稀疏模式并只分解非零元素。

## 7.11 PYPOWER 的复数矩阵实现

PYPOWER 从：

$$
S=\operatorname{diag}(V)\operatorname{conj}(Y_{\mathrm{bus}}V)
$$

直接对复电压求导：

```python
dS_dVm, dS_dVa = dSbus_dV(Ybus, V)
```

得到：

- `dS_dVa`：`∂S/∂θ`；
- `dS_dVm`：`∂S/∂Vm`。

然后：

```python
J11 = dS_dVa[pvpq][:, pvpq].real
J12 = dS_dVm[pvpq][:, pq].real
J21 = dS_dVa[pq][:, pvpq].imag
J22 = dS_dVm[pq][:, pq].imag
```

取实部和虚部：

$$
\frac{\partial(P+jQ)}{\partial\theta}
=
\frac{\partial P}{\partial\theta}
+j\frac{\partial Q}{\partial\theta}.
$$

因此：

- 实部对应 P 方程；
- 虚部对应 Q 方程。

本项目测试验证：

```python
assert np.max(np.abs(J_our - J_pypower)) < 1e-12
```

case9 实际最大差：

```text
0.0
```

## 7.12 有限差分验证

一元偏导的中心差分：

$$
\frac{\partial F_i}{\partial x_j}
\approx
\frac{F_i(x+\epsilon e_j)-F_i(x-\epsilon e_j)}{2\epsilon}.
$$

为了验证整个第 `j` 列：

```python
xp = x0.copy()
xm = x0.copy()
xp[j] += eps
xm[j] -= eps
J_numeric[:, j] = (F(xp) - F(xm)) / (2 * eps)
```

本项目使用：

$$
\epsilon=10^{-6}.
$$

`lab06` 选取若干元素验证，最大绝对误差约：

```text
1.049e-09
```

这证明解析雅可比公式和 Python 实现一致。

## 7.13 差分步长选择

- 步长太大：泰勒展开截断误差明显；
- 步长太小：两个相近浮点数相减产生舍入误差。

常用范围：

$$
10^{-4}\sim10^{-7}.
$$

有限差分适合验证，不应替代生产环境中的解析雅可比。

## 7.14 实验

```bash
.venv/bin/python powerflow-textbook/labs/lab06_finite_difference.py
```

观察：

1. 解析导数和差分导数是否一致；
2. 零元素是否仍接近零；
3. 不同位置的绝对误差量级；
4. 把 `step` 改为 `1e-4` 或 `1e-10`，观察误差变化。
