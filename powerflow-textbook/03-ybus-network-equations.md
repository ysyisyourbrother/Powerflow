# 第 3 章 节点导纳矩阵与网络方程

## 3.1 网络方程

设母线电压向量：

$$
V=
\begin{bmatrix}
V_1\\
\vdots\\
V_{n_b}
\end{bmatrix},
$$

节点导纳矩阵：

$$
Y_{\mathrm{bus}}=
\begin{bmatrix}
Y_{11} & \cdots & Y_{1n_b}\\
\vdots & \ddots & \vdots\\
Y_{n_b1} & \cdots & Y_{n_bn_b}
\end{bmatrix}.
$$

忽略并联注入时：

$$
I_{\mathrm{bus}}=Y_{\mathrm{bus}}V.
$$

母线复功率：

$$
S_{\mathrm{bus}}
=\operatorname{diag}(V)\operatorname{conj}(I_{\mathrm{bus}}).
$$

逐元素：

$$
S_i=V_i\operatorname{conj}\left(\sum_{k=1}^{n_b}Y_{ik}V_k\right).
$$

## 3.2 普通线路模型

支路 `i-k` 的串联阻抗：

$$
z_{ik}=r_{ik}+jx_{ik}.
$$

串联导纳：

$$
y_{ik}=\frac{1}{z_{ik}}
=\frac{1}{r_{ik}+jx_{ik}}.
$$

线路总充电电纳记为 `b`，通常两端各分一半：

$$
Y_{ff}=y_{ik}+j\frac{b}{2},
$$

$$
Y_{tt}=y_{ik}+j\frac{b}{2},
$$

$$
Y_{ft}=Y_{tf}=-y_{ik}.
$$

四端口贡献：

$$
\begin{bmatrix}
I_f\\
I_t
\end{bmatrix}
=
\begin{bmatrix}
Y_{ff} & Y_{ft}\\
Y_{tf} & Y_{tt}
\end{bmatrix}
\begin{bmatrix}
V_f\\
V_t
\end{bmatrix}.
$$

装配 `Ybus`：

$$
Y_{ff}\rightarrow Y_{ff},
$$

$$
Y_{tt}\rightarrow Y_{tt},
$$

$$
Y_{ft}\rightarrow Y_{ft},
$$

$$
Y_{tf}\rightarrow Y_{tf}.
$$

## 3.3 Python 构造

```python
from pypower.api import makeYbus

Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
```

返回：

| 变量 | 形状 | 含义 |
|---|---|---|
| `Ybus` | `nb × nb` | 母线导纳矩阵 |
| `Yf` | `nbr × nb` | 从首端看各线路电流 |
| `Yt` | `nbr × nb` | 从末端看各线路电流 |

`Yf` 和 `Yt` 用于最后回算线路潮流：

$$
I_f=Y_fV,\qquad I_t=Y_tV.
$$

## 3.4 手算支路 4-5

支路参数：

$$
r=0.017,\qquad x=0.092,\qquad b=0.158.
$$

串联导纳：

$$
y_{45}=\frac{1}{0.017+j0.092}.
$$

数值：

$$
y_{45}=1.942191-j10.510682.
$$

两端自导纳：

$$
Y_{ff}=y_{45}+j\frac{0.158}{2}
=1.942191-j10.431682.
$$

$$
Y_{tt}=1.942191-j10.431682.
$$

互导纳：

$$
Y_{ft}=Y_{tf}=-y_{45}
=-1.942191+j10.510682.
$$

因此：

```text
Ybus[3, 3] += Yff
Ybus[3, 4] += Yft
Ybus[4, 3] += Ytf
Ybus[4, 4] += Ytt
```

这里 Python 索引从 0 开始，母线 4 对应内部索引 3。

## 3.5 对角元和非对角元的物理意义

对于母线 `i`：

$$
Y_{ii}=G_{ii}+jB_{ii}
$$

是把所有连接到该母线的支路自导纳相加。它决定本母线电压变化对自身注入的影响。

非对角元：

$$
Y_{ik}=G_{ik}+jB_{ik},\qquad i\ne k
$$

来自直接连接母线 `i` 与 `k` 的支路。若两个母线之间没有支路，
则 `Yik=0`。

`Ybus` 的稀疏结构就是电网的邻接结构。

## 3.6 case9 的 Ybus 结构

非零连接：

```text
1-1, 1-4
2-2, 2-8
3-3, 3-6
4-1, 4-4, 4-5, 4-9
5-4, 5-5, 5-6
6-3, 6-5, 6-6, 6-7
7-6, 7-7, 7-8
8-2, 8-7, 8-8, 8-9
9-4, 9-8, 9-9
```

由于 case9 不含移相器，未归一化的电气网络满足：

$$
Y_{\mathrm{bus}}=Y_{\mathrm{bus}}^T.
$$

Python 检查：

```python
Y = Ybus.toarray()
asymmetry = np.linalg.norm(Y - Y.T, ord=np.inf)
print(asymmetry)
```

预期为 `0.0`。

## 3.7 几个典型矩阵元素

| 行 | 列 | 实部 | 虚部 |
|---:|---:|---:|---:|
| 1 | 1 | 0.00000000 | -17.36111111 |
| 1 | 4 | 0.00000000 | 17.36111111 |
| 4 | 4 | 3.30737896 | -39.30888873 |
| 4 | 5 | -1.94219125 | 10.51068205 |
| 5 | 5 | 3.22420039 | -15.84092701 |
| 8 | 9 | -1.18760438 | 5.97513453 |
| 9 | 9 | 2.55279209 | -17.33823009 |

虚部通常比实部大，这是因为高压输电线以电抗为主。

## 3.8 从 `Ybus` 计算初始 P、Q

初始电压：

$$
V_m=1,\qquad \theta=0.
$$

初始：

$$
V_iV_k=1,\qquad \theta_{ik}=0.
$$

所以：

$$
P_i=\sum_kG_{ik},\qquad Q_i=-\sum_kB_{ik}.
$$

PYPOWER 计算得到：

```text
P0 = [0, 0, 0, 0, 0, 0, 0, 0, 0]
Q4 = -0.1670
Q5 = -0.2580
Q6 = -0.2835
Q7 = -0.1790
Q8 = -0.2275
Q9 = -0.2410
```

初始平启动不为零的 Q 就是充电电纳等网络效应造成的。

## 3.9 为什么使用稀疏矩阵

`Ybus` 每个母线只与少数母线相连，多数元素为零。对于大规模电网：

- 稠密矩阵会消耗大量内存；
- 求逆和分解会浪费大量计算；
- 稀疏矩阵只保存非零元素及其位置。

PYPOWER 的 `makeYbus` 返回 SciPy 稀疏矩阵。当前 case9 使用 `.toarray()` 是为了教学展示；
生产代码应保持稀疏格式。

## 3.10 实验

```bash
.venv/bin/python powerflow-textbook/labs/lab03_build_matrices.py
```

检查：

1. 支路 4-5 是否得到预期的 `Yff/Yft`；
2. `Ybus` 是否严格对称；
3. 非对角非零元素是否与拓扑一一对应；
4. `Sbus` 是否等于发电减负荷。
