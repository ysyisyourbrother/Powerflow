# 第 10 章 从 case9 走向比赛求解器

## 10.1 当前项目已经具备的能力

- PYPOWER 标准数据读取；
- Pandapower 工程建模；
- `Ybus` 和 `Sbus` 构造；
- 极坐标牛顿法完整实现；
- 残差、雅可比、修正量逐步追踪；
- 发电机和线路结果回算；
- 数值验证和绘图层。

比赛扩展时，不需要重写数据层，而应逐步替换或增强求解层。

## 10.2 推荐目录演化

保持现有目录不变：

```text
src/powerflow/
├── cases.py             # 数据读取与格式转换
├── newton_raphson.py    # 教学和原型求解器
├── sparse_solver.py     # 待扩展：稀疏生产求解器
├── preprocess.py        # 待扩展：坏数据、网损、单位转换
├── validation.py        # 待扩展：残差、物理校验
└── plotting.py
```

新增实验继续放在 `src/examples/`，新增比赛适配代码放在 `src/powerflow/`。

## 10.3 从稠密到稀疏

当前教学代码：

```python
J = jacobian_polar(Ybus, V, pv, pq)
dx = np.linalg.solve(J, -mismatch)
```

适合理解算法，不适合上万节点。

生产版本应：

1. 用 SciPy 稀疏矩阵构造 `J`；
2. 只保存非零元素；
3. 使用 `splu` 因式分解；
4. 对重复拓扑结构进行符号分析复用；
5. 必要时使用迭代线性求解器。

PYPOWER 已提供：

```python
from pypower.newtonpf import newtonpf
```

可用它作为正确性基准。

## 10.4 数据预处理

比赛数据常见问题：

- 母线编号不连续；
- 单位不是 MW/Mvar 或 p.u.；
- 多台发电机接在同一母线；
- 支路首末端方向不一致；
- 停运设备仍保留；
- 负荷和发电机名称带空值；
- 外部文件是 CSV、JSON 或 Excel。

建议流程：

```text
原始数据
→ 字段校验
→ 单位统一
→ 编号重排
→ 缺失值处理
→ PYPOWER 字典或 Pandapower 表
→ 潮流求解
→ 结果回写原始编号
```

## 10.5 使用 Pandapower 做工程扩展

创建元件：

```python
import pandapower as pp

net = pp.create_empty_network(sn_mva=100)
b1 = pp.create_bus(net, vn_kv=110, name="bus1")
b2 = pp.create_bus(net, vn_kv=110, name="bus2")
pp.create_ext_grid(net, b1, vm_pu=1.0)
pp.create_line_from_parameters(
    net, b1, b2,
    length_km=10,
    r_ohm_per_km=0.05,
    x_ohm_per_km=0.2,
    c_nf_per_km=10,
    max_i_ka=0.5,
)
pp.create_load(net, b2, p_mw=20, q_mvar=5)
pp.runpp(net, algorithm="nr", calculate_voltage_angles=True)
```

适合比赛中的：

- 开关状态；
- 变压器分接头；
- ZIP 负荷；
- 多电压等级；
- 三相不平衡；
- 时间序列仿真。

## 10.6 使用 PYPOWER 做算法对照

PYPOWER 的优势是数据数组和经典算法清晰。可以对照：

```python
from pypower.api import newtonpf, fdpf, gausspf, ppoption
```

验证自研求解器：

- 收敛电压；
- 最大残差；
- 迭代次数；
- 平衡节点功率；
- 线路潮流。

## 10.7 性能优化方向

### 向量化

当前教学雅可比使用双重循环。大规模版本可向量化：

- 先构造 `theta`、`Vm`、`G`、`B`；
- 用广播计算所有分块；
- 用 `scipy.sparse.coo_matrix` 只装配非零位置。

### 稀疏分解

```python
from scipy.sparse.linalg import splu

lu = splu(J_csc)
dx = lu.solve(-F)
```

### 复用符号结构

若母线拓扑不变，可复用：

- 雅可比非零模式；
- 排序结果；
- 部分因式分解结构。

### 并行和 GPU

只有在单机稀疏求解成为瓶颈后才考虑 GPU。多数潮流问题首先受稀疏线性求解和
数据管理限制，而不是简单矩阵乘法。

## 10.8 AI 潮流计算比赛的可能任务

### 任务一：加速传统潮流

- 用神经网络预测初值；
- 用传统牛顿法做最终收敛；
- 保证结果满足物理残差。

### 任务二：学习雅可比近似

- 网络输出近似雅可比；
- 再以小规模解析雅可比校正；
- 重点评估泛化到不同负荷场景。

### 任务三：识别病态场景

- 判断是否接近电压崩溃；
- 预测不收敛样本；
- 输出可信度。

### 任务四：图神经网络

把母线作为节点，把支路和导纳作为边特征：

```text
节点特征：母线类型、Pd/Qd、Vm/θ、发电机限值
边特征：G/B、容量、变比、状态
```

最终仍必须用潮流方程检查模型输出。

## 10.9 物理约束不能省略

即使使用机器学习，也应检查：

$$
F(V)<\varepsilon.
$$

以及：

- 发电机有功、无功限值；
- 母线电压上下限；
- 线路容量；
- 平衡节点功率平衡；
- 数值稳定性和异常数据。

## 10.10 推荐开发路线

1. 把当前教学求解器扩到 case14、case30、case57；
2. 对比 PYPOWER 的电压、迭代和残差；
3. 改为稀疏雅可比与稀疏 LU；
4. 用 Pandapower 做批量场景和结果表；
5. 加入 CSV/JSON 数据适配；
6. 建立回归测试；
7. 再考虑神经网络初值或雅可比加速。

## 10.11 比赛提交前检查

- 固定随机种子；
- 固定依赖版本；
- 保留纯传统求解基线；
- 每个场景都要检查残差；
- 对超时、不收敛和异常输入给出明确状态；
- 不把绘图和日志混入核心求解路径；
- 结果回写到原始母线编号；
- 用至少两个独立实现交叉验证。
