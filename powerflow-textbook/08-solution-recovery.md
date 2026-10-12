# 第 8 章 结果回算、线路潮流与物理校验

## 8.1 为什么还需要回算

牛顿法最终只给复电压：

$$
V_i=V_{m,i}e^{j\theta_i}.
$$

还需要得到：

- PV 发电机无功；
- 平衡发电机有功、无功；
- 每条线路两端潮流；
- 网络损耗；
- 各母线净注入。

这些量由最终电压和网络方程重新计算。

## 8.2 母线净注入

代码：

```python
Sbus_calc = V * np.conj(Ybus.dot(V))
```

公式：

$$
S_{\mathrm{bus},i}
=V_i\operatorname{conj}\left(\sum_kY_{ik}V_k\right).
$$

实部为净有功注入：

$$
P_{\mathrm{bus},i}=\operatorname{Re}(S_{\mathrm{bus},i}),
$$

虚部为净无功注入：

$$
Q_{\mathrm{bus},i}=\operatorname{Im}(S_{\mathrm{bus},i}).
$$

这里的“净”等于发电机减负荷。

## 8.3 回算 PV 发电机无功

PV 母线电压幅值固定，但 Q 未知。得到最终 `V` 后：

$$
Q_{\mathrm{bus},i}
=Q_{g,i}-Q_{d,i}.
$$

因此：

$$
Q_{g,i}=Q_{\mathrm{bus},i}+Q_{d,i}.
$$

PYPOWER 的 `pfsoln` 使用：

```python
Sbus = V[gbus] * np.conj(Ybus[gbus, :] * V)
gen[on, QG] = Sbus.imag * baseMVA + bus[gbus, QD]
```

最终：

| 发电机 | 母线 | Qg Mvar |
|---:|---:|---:|
| 1 | 1 | 24.068958 |
| 2 | 2 | 14.460120 |
| 3 | 3 | -3.649026 |

PV 发电机 2、3 的无功不再是初值 0，而是电压控制后的结果。

## 8.4 回算平衡发电机有功

平衡母线：

$$
P_{\mathrm{bus},1}=P_{g,1}-P_{d,1}.
$$

所以：

$$
P_{g,1}=P_{\mathrm{bus},1}+P_{d,1}.
$$

PYPOWER 代码：

```python
gen[on[refgen[0]], PG] = (
    Sbus[refgen[0]].real * baseMVA + bus[ref, PD]
)
```

当前结果：

$$
P_{g,1}=71.954702\ \mathrm{MW}.
$$

这个值吸收了：

- 发电机 2 的 `163 MW`；
- 发电机 3 的 `85 MW`；
- 总负荷 `315 MW`；
- 线路有功损耗 `4.954702 MW`。

## 8.5 功率平衡校验

总发电：

$$
P_{\mathrm{gen}}
=71.954702+163+85
=319.954702\ \mathrm{MW}.
$$

总负荷：

$$
P_{\mathrm{load}}=315\ \mathrm{MW}.
$$

网络有功损耗：

$$
P_{\mathrm{loss}}
=P_{\mathrm{gen}}-P_{\mathrm{load}}
=4.954702\ \mathrm{MW}.
$$

也可以直接从所有母线注入求和：

$$
P_{\mathrm{loss}}
=\sum_iP_{\mathrm{bus},i}.
$$

数值误差范围内应与上式一致。

## 8.6 线路首末两端功率

PYPOWER 的 `makeYbus` 还返回：

```python
Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
```

首端电流：

$$
I_f=Y_fV.
$$

末端电流：

$$
I_t=Y_tV.
$$

首端复功率：

$$
S_f=V_f\operatorname{conj}(I_f)S_{\mathrm{base}}.
$$

末端复功率：

$$
S_t=V_t\operatorname{conj}(I_t)S_{\mathrm{base}}.
$$

代码：

```python
Sf = V[F_BUS] * np.conj(Yf * V) * baseMVA
St = V[T_BUS] * np.conj(Yt * V) * baseMVA
branch[:, [PF, QF, PT, QT]] = np.c_[
    Sf.real, Sf.imag, St.real, St.imag
]
```

## 8.7 case9 线路潮流

| 首端 | 末端 | PF MW | QF Mvar | PT MW | QT Mvar | 容量 MVA |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 71.954702 | 24.068958 | -71.954702 | -20.753045 | 250 |
| 4 | 5 | 30.728280 | -0.585851 | -30.554686 | -13.687950 | 250 |
| 5 | 6 | -59.445314 | -16.312050 | 60.893866 | -12.427470 | 150 |
| 3 | 6 | 85.000000 | -3.649026 | -85.000000 | 7.890678 | 300 |
| 6 | 7 | 24.106134 | 4.536791 | -24.010648 | -24.400762 | 150 |
| 7 | 8 | -75.989352 | -10.599238 | 76.495564 | 0.256239 | 250 |
| 8 | 2 | -163.000000 | 2.276190 | 163.000000 | 14.460120 | 250 |
| 8 | 9 | 86.504436 | -2.532429 | -84.039887 | -14.281983 | 250 |
| 9 | 4 | -40.960113 | -35.718017 | 41.226421 | 21.338895 | 250 |

线路损耗：

$$
P_{\mathrm{loss},ik}=P_f+P_t.
$$

例如，母线 4-5：

$$
P_{\mathrm{loss},45}
=30.728280-30.554686
=0.173594\ \mathrm{MW}.
$$

所有线路损耗之和：

$$
\sum P_{\mathrm{loss},ik}=4.954702\ \mathrm{MW}.
$$

## 8.8 线路负载率

若长期容量为 `RATE_A`，负载率：

$$
\mathrm{loading}_{ik}
=
100\%\cdot
\frac{\max(|S_f|,|S_t|)}{\mathrm{RATE_A}}.
$$

这是仿真图使用的主要视觉变量。

## 8.9 母线电压结果

| 母线 | Vm p.u. | Va 度 | 负荷 P MW | 负荷 Q Mvar |
|---:|---:|---:|---:|---:|
| 1 | 1.000000000000 | 0.000000000000 | 0 | 0 |
| 2 | 1.000000000000 | 9.668741126628 | 0 | 0 |
| 3 | 1.000000000000 | 4.771073237177 | 0 | 0 |
| 4 | 0.987006852392 | -2.406643919519 | 0 | 0 |
| 5 | 0.975472177085 | -4.017264326708 | 90 | 30 |
| 6 | 1.003375436453 | 1.925601686829 | 0 | 0 |
| 7 | 0.985644881725 | 0.621544555389 | 100 | 35 |
| 8 | 0.996185245809 | 3.799120192692 | 0 | 0 |
| 9 | 0.957621040430 | -4.349933576561 | 125 | 50 |

最低电压：

$$
V_{m,\min}=0.957621040430\ \mathrm{p.u.}
$$

出现在母线 9。

## 8.10 发电机结果

| 发电机 | 母线 | Pg MW | Qg Mvar |
|---:|---:|---:|---:|
| 1 | 1 | 71.954701589222 | 24.068957772759 |
| 2 | 2 | 163.000000000000 | 14.460119531126 |
| 3 | 3 | 85.000000000000 | -3.649025534210 |

发电机总无功：

$$
Q_{\mathrm{gen}}
=24.068958+14.460120-3.649026
=34.880051\ \mathrm{Mvar}.
$$

负荷无功：

$$
Q_{\mathrm{load}}=115\ \mathrm{Mvar}.
$$

两者相减对应线路充电和网络无功损耗等效应。

## 8.11 自研、PYPOWER、Pandapower 对照

运行：

```bash
.venv/bin/python src/examples/compare_backends.py
```

当前最大电压差：

| 对照 | ΔVm | ΔVa 度 |
|---|---:|---:|
| 自研 vs PYPOWER | 2.220e-16 | 1.066e-14 |
| 自研 vs Pandapower | 3.331e-16 | 6.217e-15 |

这属于浮点舍入误差量级。

## 8.12 结果校验清单

1. 自研电压与 PYPOWER 是否一致；
2. 自研电压与 Pandapower 是否一致；
3. 所有残差是否小于容差；
4. 发电机总出力是否等于负荷加损耗；
5. 所有线路首末端功率之和是否为网络损耗；
6. 平衡发电机是否吸收了系统不平衡；
7. 电压幅值是否在合理范围；
8. 线路负载率是否超限。

## 8.13 实验

```bash
.venv/bin/python src/examples/step1_case9.py
.venv/bin/python src/examples/step2_pandapower.py
.venv/bin/python src/examples/compare_backends.py
```

检查图片：

```text
src/examples/case9_result.png
```

图中左侧是线路负载和母线类型，右侧是母线电压幅值。
