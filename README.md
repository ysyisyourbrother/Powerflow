# Powerflow

电力系统潮流计算学习与仿真项目，全部使用 Python。

核心工具链：

- **PYPOWER 5.1.21**：提供标准算例、母线/发电机/支路数据结构、导纳矩阵、
  功率注入和成熟牛顿法参考实现。
- **Pandapower 3.5.6**：提供工程化建模表、网络转换、潮流求解以及母线、发电机、
  线路结果表。
- **项目自研教学求解器**：把极坐标牛顿-拉夫逊法的残差、雅可比、线性修正和状态
  更新完整展开，供逐行学习与验证。

## 目录结构

```text
Powerflow/
├── src/
│   ├── powerflow/              # Python 包：数据、牛拉法、回算、绘图
│   │   ├── cases.py
│   │   ├── newton_raphson.py
│   │   ├── plotting.py
│   │   └── report.py
│   └── examples/
│       ├── smoke_test.py       # 环境与数值结果自检
│       ├── step1_case9.py      # 自研牛顿法求解 case9
│       ├── step2_pandapower.py # Pandapower 建模与仿真
│       ├── step3_case39.py     # IEEE 39 三路径完整教程
│       └── compare_backends.py # 三个求解路径对照
├── dataset/                    # 比赛数据，目录约定保持不变
├── doc/                        # 算例详解和核心代码剖析
├── powerflow-textbook/         # 从数学到 Python 实现的完整教材
│   └── labs/                   # 8 个 Python 实验
├── tests/                      # 自动测试
├── install.sh                  # 创建 .venv 并安装依赖
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 安装

要求 Python 3.11 或 3.12。

```bash
./install.sh
```

脚本会：

1. 在项目根目录创建 `.venv`；
2. 安装 PYPOWER、Pandapower、NumPy、SciPy、Matplotlib；
3. 以 editable 模式安装本项目；
4. 自动运行环境自检。

也可以手动安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

## 快速验证

```bash
.venv/bin/python src/examples/smoke_test.py
```

预期结果：

```text
自研牛拉法: 收敛=True, 迭代=4
PYPOWER   : 收敛= True
Pandapower: 收敛= True , 迭代= 4
[PASS] Python 潮流计算环境正常
```

## 跑第一个例程

```bash
.venv/bin/python src/examples/step1_case9.py
```

程序会：

1. 从 PYPOWER 读取 IEEE 9 节点算例；
2. 构造 `Ybus` 和计划功率注入 `Sbus`；
3. 用自研 Python 牛顿法迭代 4 次；
4. 打印每次迭代的残差和修正量摘要；
5. 回算发电机出力和线路潮流；
6. 保存 `src/examples/case9_result.png`。

## Pandapower 仿真

```bash
.venv/bin/python src/examples/step2_pandapower.py
```

该脚本把同一个 PYPOWER `case9` 转成 Pandapower 网络，再调用：

```python
pandapower.runpp(
    net,
    algorithm="nr",
    calculate_voltage_angles=True,
    tolerance_mva=1e-10,
)
```

最后打印 `res_bus`、`res_gen`、`res_ext_grid` 和 `res_line`。

## 已有 IEEE 39 节点数据

`dataset/ieee39/` 已整理好 IEEE 39 节点 New England 系统原始数据和参考解：

- PYPOWER：39 母线、10 发电机、46 支路；
- Pandapower：39 母线、21 负荷、35 线路、11 台变压器；
- 原始 CSV、JSON、NPZ、参考潮流结果和 SHA-256 校验；
- 说明入口：[dataset/ieee39/README.md](dataset/ieee39/README.md)。

## IEEE 39 节点完整教程

```bash
.venv/bin/python src/examples/step3_case39.py
```

一个算例，三条求解路径互相验证：

| 路径 | 实现 | 迭代次数 |
|---|---|---:|
| A | PYPOWER `runpf` | 1（算例初值已接近收敛解） |
| B | 自研极坐标牛顿法 | 1；平启动 4 |
| C | Pandapower `runpp` | 4 |

三个实现的最大电压差异：自研 vs PYPOWER 为 `2.2e-16` pu，
Pandapower vs PYPOWER 为 `2.9e-14` pu。

详细教程（数据、求解方法、结果解读）：[doc/case39/README.md](doc/case39/README.md)
—— 分 10 个文件，每个 67 ~ 225 行，方便手机阅读。

配套数据导出脚本：

```bash
.venv/bin/python dump_case39_tables.py     # 原始数据表
.venv/bin/python dump_case39_solution.py   # 潮流结果表
```

## 三个结果路径

```bash
.venv/bin/python src/examples/compare_backends.py
```

| 路径 | 目的 |
|---|---|
| `powerflow.solve_case9()` | 教学求解器，完整可观察每次迭代 |
| `powerflow.run_pypower_case9()` | PYPOWER 成熟参考实现 |
| `powerflow.run_pandapower_case9()` | Pandapower 工程化建模和结果表 |

当前 case9 结果中，自研求解器与两个成熟库的母线电压最大差异约为 `1e-15`。

## 教材

入口：[powerflow-textbook/README.md](powerflow-textbook/README.md)

建议顺序：

1. 第 1-4 章：算例、数据、标幺值、`Ybus`、功率平衡和未知量；
2. 第 5 章：完整牛顿迭代数值过程；
3. 第 6 章：Python 代码与公式逐行对照；
4. 第 7 章：解析雅可比与有限差分验证；
5. 第 8-10 章：结果回算、实验和比赛扩展。

运行全部实验：

```bash
for f in powerflow-textbook/labs/lab*.py; do
  .venv/bin/python "$f"
done
```

## 测试

```bash
.venv/bin/python -m pytest
```

测试覆盖：

- case9 是否 4 次收敛；
- 自研极坐标雅可比是否与 PYPOWER 复数导数矩阵一致；
- 解析雅可比是否通过中心差分；
- 自研、PYPOWER、Pandapower 电压结果是否一致。

## 环境清理

只删除 Python 虚拟环境：

```bash
rm -rf .venv
```

依赖和版本声明保留在 `requirements.txt` 与 `pyproject.toml` 中，可随时重建。
