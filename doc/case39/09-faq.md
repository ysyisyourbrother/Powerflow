# 第五部分　常见问题与排错

> 10 个初学者最常遇到的问题

### Q1：为什么只迭代 1 次？是不是没算？

不是。见 1.11 节 ① 和 2.9 节。这个算例自带的初值已经接近收敛解，
初值处失配只有 2.91e-05 pu。想看完整迭代做平启动实验（2.10 节）。

### Q2：`result` 里怎么没有 `iterations`？

PYPOWER 的 `runpf` 返回值确实不含这个字段。迭代次数只在 `VERBOSE=1` 时打印。
想程序化获取可以：

- 用自研求解器（`solve_newton_raphson` 返回 `result.iterations`）；
- 用 Pandapower（`net._ppc["iterations"]`）；
- 自己在循环里调 `pypower.newtonpf`。

### Q3：`calculate_voltage_angles` 是什么？

Pandapower 的 `runpp` **默认不计算电压相角**（`calculate_voltage_angles=False`），
因为配电网络通常不需要相角。但输电网分析必须要，所以一定要显式设成 `True`，
否则 `res_bus.va_degree` 会全是 0，你会以为算错了。

### Q4：为什么电压结果超过了 1.06 pu 的上限？

母线 36 是 1.0636 pu，确实超了。**潮流计算不处理约束**，它只解方程。
数据里的 `VMAX`/`VMIN` 只是记录，求解器不会强制。
要做带约束的优化，用最优潮流（OPF）。

### Q5：无功损耗是负的，我算错了吗？

没有。见 3.4 节。345 kV 线路的充电电容发出的无功（1112.75 Mvar）超过了
串联电抗消耗的无功（1224.92 Mvar 中的相抵部分），所以网络净"发出"无功
112.16 Mvar。这是超高压电网的正常现象。

### Q6：告警 `Matplotlib is not a writable directory`

matplotlib 想把字体缓存写进 `~/.matplotlib` 但没权限。解决办法：

```bash
export MPLCONFIGDIR=/tmp/mpl_cache
```

### Q7：图里中文显示成方块

matplotlib 默认字体不含中文。本项目脚本已经自动配置了
`PingFang SC` 等 macOS 自带中文字体。如果你自己写脚本，加上：

```python
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
```

### Q8：`PF_ALG` 该选什么？

| 值 | 算法 | 何时用 |
|---:|---|---|
| 1 | 牛顿法（极坐标） | **默认，首选**。收敛快、迭代少 |
| 2 / 3 | 快速解耦法 | 大规模系统、内存受限时；迭代多但每步便宜 |
| 4 | 高斯-赛德尔 | 只用于教学理解，实际不用 |

初学者**一直用 1** 就行。

### Q9：改了负荷，结果会怎么变？

改负荷 → 平衡机的 `Pg`、`Qg` 会变（兜底），各母线电压会变。
**PV 节点的 `Pg` 不会变**（它是设定值），但 `Qg` 会变。

### Q10：单位怎么换算？

全系统 `baseMVA = 100`。所以：

```text
标幺值 × 100 = MW / Mvar / MVA
例如：ΣPg = 62.978711 pu × 100 = 6297.8711 MW
```

**PYPOWER 内部计算全用标幺值，只有 `bus`、`gen`、`branch` 表里的
`PD`、`QD`、`PG`、`QG`、`RATE_A` 等是 MW/Mvar/MVA。**


---

[← 完整代码与运行](08-code.md)　|　[返回目录](README.md)　|　[下一步学什么 →](10-next-steps.md)
