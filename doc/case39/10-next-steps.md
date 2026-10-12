# 第六部分　下一步学什么

> 巩固实验、配套教材、数据集、深入方向

## 6.1 巩固：亲手做这几个实验

1. **改负荷重算**：把母线 20 的负荷（680 MW）加倍，看平衡机出力和
   哪些线路负载率飙升。预期：母线 20 附近的支路（19-20、20-34）负载率大涨。

2. **改算法对比**：把 `PF_ALG` 改成 2、3（快速解耦法），比较迭代次数。
   预期：快速解耦法迭代次数更多，但每次迭代更快。

3. **断开一条线路**：把母线 16→19 这条负载率最高的支路（第 27 条）
   `BR_STATUS` 设为 0，看潮流如何重新分布、是否有线路过载。
   这是 N-1 校验的雏形。

4. **松弛容差**：把 `PF_TOL` 从 1e-10 改成 1e-3，看结果差多少，
   理解"容差"对结果精度的影响。

5. **自己写一次雅可比**：参照 `src/powerflow/newton_raphson.py`，
   用有限差分验证解析雅可比（项目里的 `tests/test_finite_difference.py` 就是干这个的）。

## 6.2 本项目的配套教材

当前项目已经有一套完整的 case9 教材，建议按顺序读：

| 文件 | 内容 |
|---|---|
| [powerflow-textbook/00-learning-map.md](../../powerflow-textbook/00-learning-map.md) | 学习地图 |
| [powerflow-textbook/02-data-model-and-per-unit.md](../../powerflow-textbook/02-data-model-and-per-unit.md) | 数据模型与标幺值 |
| [powerflow-textbook/03-ybus-network-equations.md](../../powerflow-textbook/03-ybus-network-equations.md) | Ybus 与网络方程 |
| [powerflow-textbook/04-power-balance-and-unknowns.md](../../powerflow-textbook/04-power-balance-and-unknowns.md) | 功率平衡与未知量 |
| [powerflow-textbook/05-newton-raphson.md](../../powerflow-textbook/05-newton-raphson.md) | 牛顿-拉夫逊法（重点） |
| [powerflow-textbook/07-jacobian.md](../../powerflow-textbook/07-jacobian.md) | 雅可比矩阵推导 |
| [powerflow-textbook/08-solution-recovery.md](../../powerflow-textbook/08-solution-recovery.md) | 结果回算 |
| [doc/case9-guide.md](../case9-guide.md) | IEEE 9 节点详解（更简单的入门算例） |
| [doc/powerflow-newton.md](../powerflow-newton.md) | 自研牛顿法代码剖析 |

**推荐路径**：先用 case9 弄懂原理（3 机 9 节点，看得清），
再用本教程的 case39 练手（10 机 39 节点，接近真实电网规模）。

## 6.3 数据集

本项目的 IEEE 39 节点数据已整理归档在
[dataset/ieee39/README.md](../../dataset/ieee39/README.md)，包含：

- 原始 CSV、JSON、NPZ（PYPOWER 与 Pandapower 两套表示）；
- 参考潮流结果（母线、发电机、支路潮流）；
- SHA-256 校验和，可验证数据未被改动。

当你想脱离 Python 库、直接用 CSV 读数据做题时，用这个目录。

## 6.4 继续深入的方向

| 方向 | 关键词 |
|---|---|
| 最优潮流 OPF | `pypower.api.runopf`、`gencost`、约束处理 |
| 无功限值处理 | PV/PQ 转换、外层循环 |
| 连续潮流 | 电压稳定、PV 曲线、负荷裕度 |
| 暂态稳定 | 发电机模型、故障仿真（case39 的原始用途！） |
| 状态估计 | 量测冗余、加权最小二乘 |
| N-1 安全校验 | 断线分析、越限筛查 |


---

[← 常见问题与排错](09-faq.md)　|　[返回目录](README.md)
