# 第 0 章 学习地图

## 0.1 一条完整计算链

潮流计算不是简单地“调用求解器”，而是一条连续的数据流：

$$
\text{母线/支路数据}
\rightarrow \text{导纳矩阵}
\rightarrow \text{计划功率注入}
\rightarrow \text{非线性残差}
\rightarrow \text{雅可比线性化}
\rightarrow \text{迭代修正}
\rightarrow \text{收敛电压}
\rightarrow \text{物理量回算}.
$$

本教材用 Python 把链上的每一环都展开。

## 0.2 两个成熟库的分工

### PYPOWER

适合：

- 读取标准算例；
- 使用 `BUS_I`、`GEN_BUS`、`BR_R` 等常量理解原始数据结构；
- 构造 `Ybus`、`Sbus`；
- 调用成熟 Newton、快速解耦或 Gauss-Seidel 求解；
- 作为自研求解器的参考结果。

本项目核心函数：

```python
from pypower.api import case9, ext2int, makeYbus, makeSbus, bustypes

ppc = case9()
ppci = ext2int(ppc)
Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
Sbus = makeSbus(ppci["baseMVA"], ppci["bus"], ppci["gen"])
ref, pv, pq = bustypes(ppci["bus"], ppci["gen"])
```

### Pandapower

适合：

- 用表结构创建母线、负荷、发电机和线路；
- 工程化地维护元件索引与结果表；
- `runpp` 求解并直接得到 `res_bus`、`res_line` 等结果；
- 后续扩展变压器、开关、三相潮流和优化建模。

本项目通过：

```python
from pandapower.converter.pypower.from_ppc import from_ppc
import pandapower as pp

net = from_ppc(ppc, f_hz=60)
pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
```

把两个生态连接起来。

### 自研求解器

`src/powerflow/newton_raphson.py` 不追求替代成熟库，而是提供：

- 显式 P/Q 公式；
- 显式四个雅可比分块；
- 完整残差、`J`、`dx`、更新后电压轨迹；
- 与 PYPOWER 复数导数实现逐项验证的机会。

## 0.3 本教材的中心问题

每个章节都围绕以下问题展开：

1. case9 有哪些母线、发电机、负荷和线路？
2. 哪种量是已知的，哪种量是未知的？
3. `Ybus` 的每个元素从哪里来？
4. `Sbus` 为什么有正负号？
5. 为什么只选 14 个残差方程？
6. `J` 的四个分块分别是什么？
7. 为什么解 `Jdx=-F` 就能改善非线性方程？
8. 4 次迭代为什么能把残差从 `10^0` 降到 `10^-14`？
9. 怎样从电压回算发电机和线路功率？
10. 怎样证明自己的求解器没有写错？

## 0.4 建议学习节奏

| 阶段 | 章节 | 实验 | 目标 |
|---|---|---|---|
| 认识问题 | 1-2 | lab01, lab02 | 知道数据表在描述什么 |
| 建立模型 | 3-4 | lab03 | 能构造 `Ybus`、`Sbus`、`F` |
| 掌握算法 | 5 | lab04, lab05 | 能手工完成一次迭代 |
| 对照代码 | 6-7 | lab04, lab06 | 公式和 Python 一一对应 |
| 验证结果 | 8 | lab01, lab05 | 能解释电压、出力和损耗 |
| 扩展能力 | 9-10 | lab07, lab08 | 能修改算例并比较算法 |

## 0.5 每章的推荐动作

1. 先读公式，不急着看代码。
2. 运行章节实验，观察数组维度和数值。
3. 回到代码，标出公式对应的每一行。
4. 修改一个输入，例如母线 9 负荷。
5. 预测结果变化，再运行验证。
6. 用有限差分或 PYPOWER 检查自己的实现。

## 0.6 完成标准

如果能够不看答案写出以下伪代码，说明已经掌握核心：

```text
V = V0
F = mismatch(V)
while not converged:
    J = jacobian(V)
    dx = solve(J, -F)
    V = update_voltage(V, dx)
    F = mismatch(V)
```

并且能够进一步说明：

- `mismatch` 中保留哪些母线的 P/Q；
- `jacobian` 的四个分块和维度；
- `update_voltage` 对 PV、PQ、平衡母线分别做什么；
- 为什么最终还要回算平衡节点功率和线路损耗。
