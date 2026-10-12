# 第 4 章 功率平衡、残差与未知向量

## 4.1 复功率注入公式

由网络方程：

$$
I_{\mathrm{bus}}=Y_{\mathrm{bus}}V.
$$

母线复功率：

$$
S_{\mathrm{bus}}
=\operatorname{diag}(V)\operatorname{conj}(I_{\mathrm{bus}}).
$$

PYPOWER 和本项目代码：

```python
S_calc = V * np.conj(Ybus.dot(V))
```

数学上：

$$
S_{\mathrm{calc},i}
=V_i\operatorname{conj}\left(\sum_kY_{ik}V_k\right).
$$

写成极坐标实部、虚部：

$$
P_i=V_i\sum_kV_k
\left(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik}\right),
$$

$$
Q_i=V_i\sum_kV_k
\left(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik}\right).
$$

## 4.2 指定注入

PYPOWER 中：

```python
Sbus = makeSbus(baseMVA, bus, gen)
```

核心关系：

$$
S_{\mathrm{spec},i}=S_{g,i}-S_{d,i}.
$$

对于恒定功率负荷：

$$
S_{d,i}=\frac{P_{d,i}+jQ_{d,i}}{S_{\mathrm{base}}}.
$$

因此母线 5、7、9：

$$
S_{\mathrm{spec},5}=-(0.90+j0.30),
$$

$$
S_{\mathrm{spec},7}=-(1.00+j0.35),
$$

$$
S_{\mathrm{spec},9}=-(1.25+j0.50).
$$

## 4.3 复残差

定义：

$$
\Delta S_i
=S_{\mathrm{calc},i}-S_{\mathrm{spec},i}.
$$

代码：

```python
delta_S = S_calc - Sbus
```

如果收敛：

$$
\Delta S_i=0,\qquad i=1,\ldots,9.
$$

但这 18 个实方程不能全部直接放入牛顿法。

## 4.4 哪些方程可以保留

平衡母线：

- `P` 未知；
- `Q` 未知。

所以不能用“计算 P 等于指定 P”或“计算 Q 等于指定 Q”作为方程。

PV 母线：

- `P` 指定；
- `Q` 未知。

所以只能使用 P 方程。

PQ 母线：

- `P`、`Q` 都指定。

所以 P、Q 方程都可以使用。

结论：

$$
F(x)=
\begin{bmatrix}
P_{\mathrm{calc},PV\cup PQ}-P_{\mathrm{spec},PV\cup PQ}\\
Q_{\mathrm{calc},PQ}-Q_{\mathrm{spec},PQ}
\end{bmatrix}.
$$

Python：

```python
F = np.r_[
    delta_S[pv].real,
    delta_S[pq].real,
    delta_S[pq].imag,
]
```

## 4.5 状态向量

状态变量由三个部分组成：

$$
x=
\begin{bmatrix}
x_{\theta,PV}\\
x_{\theta,PQ}\\
x_{V_m,PQ}
\end{bmatrix}.
$$

### PV 相角

$$
\theta_2,\theta_3.
$$

### PQ 相角

$$
\theta_4,\theta_5,\theta_6,\theta_7,\theta_8,\theta_9.
$$

### PQ 幅值

$$
V_{m,4},V_{m,5},V_{m,6},V_{m,7},V_{m,8},V_{m,9}.
$$

因此：

$$
x\in\mathbb{R}^{14}.
$$

完整状态：

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
\end{bmatrix}.
$$

## 4.6 为什么平衡节点相角固定

如果所有母线相角同时增加常数 `c`：

$$
\theta_i'=\theta_i+c,
$$

那么线路两端相角差：

$$
\theta_i'-\theta_k'
=(\theta_i+c)-(\theta_k+c)
=\theta_i-\theta_k.
$$

线路潮流只依赖相角差，因此潮流不变。为了消除这个自由度，必须固定一个参考相角。
case9 取平衡母线 1：

$$
\theta_1=0.
$$

## 4.7 方程数量核对

P 方程：

$$
|PV|+|PQ|=2+6=8.
$$

Q 方程：

$$
|PQ|=6.
$$

总方程：

$$
8+6=14.
$$

未知数：

$$
|PV|+|PQ|+|PQ|=2+6+6=14.
$$

因此每一步求解：

$$
J\in\mathbb{R}^{14\times14},
\qquad
\Delta x\in\mathbb{R}^{14},
\qquad
F\in\mathbb{R}^{14}.
$$

## 4.8 初始残差

初始 `Vm=1, θ=0` 时：

| 行 | 标签 | 数值 |
|---:|---|---:|
| 1 | ΔP2 | -1.630000000000 |
| 2 | ΔP3 | -0.850000000000 |
| 3 | ΔP4 | -2.220446049250e-16 |
| 4 | ΔP5 | 0.900000000000 |
| 5 | ΔP6 | 0.000000000000 |
| 6 | ΔP7 | 1.000000000000 |
| 7 | ΔP8 | -2.220446049250e-16 |
| 8 | ΔP9 | 1.250000000000 |
| 9 | ΔQ4 | -0.167000000000 |
| 10 | ΔQ5 | 0.042000000000 |
| 11 | ΔQ6 | -0.283500000000 |
| 12 | ΔQ7 | 0.171000000000 |
| 13 | ΔQ8 | -0.227500000000 |
| 14 | ΔQ9 | 0.259000000000 |

无穷范数：

$$
\|F_0\|_\infty
=\max_i|F_{0,i}|
=1.630000.
$$

## 4.9 为什么要用无穷范数

收敛条件：

$$
\|F\|_\infty<\varepsilon.
$$

无穷范数取最大绝对值，其含义最直观：

> 所有 P/Q 失配中，最严重的一项必须小于容差。

当前项目使用：

$$
\varepsilon=10^{-10}.
$$

## 4.10 电流平衡和功率平衡

本教材使用功率平衡、极坐标形式：

$$
F(x)=
\begin{bmatrix}
\Delta P(x)\\
\Delta Q(x)
\end{bmatrix}.
$$

还可以使用：

- 电流平衡；
- 直角坐标；
- 混合坐标。

它们都遵循：

$$
F(x)=0,\qquad
J(x)\Delta x=-F(x),
$$

区别在于残差定义与雅可比形式。

## 4.11 实验

```bash
.venv/bin/python powerflow-textbook/labs/lab03_build_matrices.py
.venv/bin/python powerflow-textbook/labs/lab04_first_iteration.py
```

检查：

1. `F0` 是否正好有 14 个元素；
2. `J0` 是否正好是 `14×14`；
3. `dx` 是否被分成 `2+6+6` 三段；
4. 第一次更新后最大残差是否从 `1.63` 降到约 `0.167`。
