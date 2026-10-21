# Powerflow

电力系统潮流计算学习与仿真项目，全部使用 Python。当前重点是：

> 按比赛格式生成 IEEE 39 节点潮流数据集，并用它训练/评测 GNN 潮流预测模型。

核心工具链：

- **PYPOWER 5.1.21**：提供标准算例、母线/发电机/支路数据结构、导纳矩阵、
  功率注入和成熟牛顿法参考实现。
- **项目自研教学求解器**：把极坐标牛顿-拉夫逊法的残差、雅可比、线性修正和状态
  更新完整展开，供逐行学习与验证。
- **赛事对齐数据集 `competition_v1`**：输入是带掩码的节点净注入 P/Q 和
  电压幅值 V，输出是全部节点 Vm 和全部支路 Δθ/Pf/Pt/Qf/Qt。

> 本项目只使用 PYPOWER。pandapower 已彻底移除，不要重新引入。

## 目录结构

```text
Powerflow/
├── src/
│   ├── powerflow/                  # Python 包
│   │   ├── cases.py                # 算例准备、平启动、内外编号映射
│   │   ├── newton_raphson.py       # 自研极坐标牛顿法
│   │   ├── dataset/
│   │   │   ├── competition.py      # 赛事格式：schema / 掩码 / 打标 / 读写
│   │   │   └── sampler.py          # 负荷扰动采样
│   │   ├── model/
│   │   │   ├── data.py             # 归一化、图张量、划分
│   │   │   ├── gnn.py              # 边感知 GNN
│   │   │   └── training.py         # 损失、训练、评估
│   │   └── ...
│   └── examples/
│       ├── build_competition_dataset.py  # 生成赛事对齐数据
│       ├── train_gnn.py                  # 训练 GNN
│       ├── evaluate_gnn.py               # 重新评估 checkpoint
│       ├── step1_case9.py                # case9 入门
│       ├── step3_case39.py               # IEEE 39 两路求解对照
│       ├── smoke_test.py                 # 环境自检
│       └── compare_backends.py           # 两个求解路径对照
├── dataset/
│   ├── README.md                   # 数据与文档总入口
│   └── ieee39/
│       ├── raw/                    # 原始 CSV/JSON/NPZ
│       ├── reference/              # PYPOWER 参考潮流结果
│       ├── processed/
│       │   └── competition_v1/     # 赛事对齐数据集（当前主用）
│       └── scripts/                # 原始数据导出脚本
├── doc/
│   ├── README.md                   # 文档索引
│   ├── case39/                     # IEEE 39 系统 10 章教程
│   └── powerflow-textbook/         # 从数学到代码的系统教材
├── tests/                          # 自动测试
├── install.sh                      # 创建 .venv 并安装依赖
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
2. 安装 PYPOWER、NumPy、SciPy、Matplotlib；
3. 以 editable 模式安装本项目；
4. 自动运行环境自检。

也可以手动安装：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps --no-build-isolation
```

如果已经使用 conda 环境（例如 `powerflow`），激活后直接用 `python` 即可，
不必再建 `.venv`。

训练 GNN 还需要 PyTorch。RTX 5090 需要 CUDA 12.8 及以上的 wheel：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

其他 GPU 按 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)
选择对应版本。也可以安装可选依赖：

```bash
pip install -e '.[ml]'
```

## 快速验证

```bash
# 使用 install.sh 创建的虚拟环境
.venv/bin/python src/examples/smoke_test.py

# 或使用已激活的 conda 环境
python src/examples/smoke_test.py
```

预期结果：

```text
PYPOWER   : 5.1.21
自研牛拉法: 收敛=True, 迭代=4, ||F||inf=5.396e-14
PYPOWER   : 收敛= True
[PASS] Python 潮流计算环境正常
```

## 生成赛事对齐数据集

```bash
# 2000 条样本，mixed 掩码（70% structural + 20% dropout + 10% extra）
python src/examples/build_competition_dataset.py --n-samples 2000

# 只生成标准潮流掩码（信息完备的基准样本）
python src/examples/build_competition_dataset.py --n-samples 2000 --mask-mode structural
```

输出目录：`dataset/ieee39/processed/competition_v1/`。

生成脚本自带硬断言：

- 基准运行方式复现 PYPOWER 参考解；
- 34 条线路 + 12 条变压器支路，B/2 与原始 `BR_B` 一致；
- structural 掩码的已知数量为 P 38 / Q 29 / V 10；
- 已知输入与真实净注入/电压一致；
- `Σ(Pf+Pt) = 网损`；
- 2000/2000 收敛；
- 写盘后读回逐字段一致。

数据集字段、单位、统计量和局限，见
[`dataset/ieee39/processed/competition_v1/README.md`](dataset/ieee39/processed/competition_v1/README.md)。

读取示例：

```python
from powerflow.dataset import CompetitionDataset

ds = CompetitionDataset("dataset/ieee39/processed/competition_v1")
item = ds[0]
item["node_features"]          # [39,3]  [P, Q, V]
item["node_masks"]             # [39,3]  [mask_P, mask_Q, mask_V]
item["branch_features"]        # [46,3]  [R, X, B/2]
item["targets"]["vm_pu"]       # [39]
item["targets"]["pf_mw"]       # [46]
```

## 训练 GNN

第一版模型是边感知消息传递网络，输入是带掩码的节点 P/Q/V 和支路
R/X/B2 + 类型，输出是全部节点 Vm 和全部支路 Δθ/Pf/Pt/Qf/Qt。
变压器变比按 1 处理，不进入模型。

```bash
# 正常训练（有 GPU 时自动用 cuda）
python src/examples/train_gnn.py --device cuda --epochs 150

# 快速冒烟
python src/examples/train_gnn.py --device cuda \
    --max-train-samples 64 --max-val-samples 32 --max-test-samples 32 \
    --epochs 5

# 重新评估
python src/examples/evaluate_gnn.py --checkpoint runs/gnn_v1/best.pt --device cuda
```

产物在 `runs/gnn_v1/`：`best.pt`、`last.pt`、`normalizer.json`、
`split.json`、`history.json`、`metrics.json`、`config.json`。

当前 150 轮训练在测试集上的误差（199 条样本）：

| 输出 | 范围 | MAE | RMSE |
|---|---|---:|---:|
| Vm | 待预测节点 | 0.00088 p.u. | 0.00126 p.u. |
| Δθ | 全部支路 | 0.0875 度 | 0.1972 度 |
| Pf | 全部支路 | 9.38 MW | 16.85 MW |
| Pt | 全部支路 | 9.38 MW | 16.85 MW |
| Qf | 全部支路 | 2.25 MVar | 3.37 MVar |
| Qt | 全部支路 | 2.49 MVar | 4.24 MVar |

这些数字只对应“单一拓扑、只扰动负荷”的 `competition_v1`，不能直接
当成比赛最终成绩。模型细节、损失、划分和消融计划见
[`dataset/model-design.md`](dataset/model-design.md)。

## IEEE 39 节点系统

`dataset/ieee39/` 已整理好 IEEE 39 节点 New England 系统的原始数据和参考解：

- PYPOWER：39 母线、10 发电机、46 支路（34 条线路 + 12 条变压器等值支路）；
- 原始 CSV、JSON、NPZ、参考潮流结果和 SHA-256 校验；
- 赛事对齐数据集和早期研究格式各一份。

入口：[dataset/ieee39/README.md](dataset/ieee39/README.md)。

## 文档

| 文档 | 内容 |
|---|---|
| [dataset/README.md](dataset/README.md) | **数据集总入口**：数据布局、字段约定、生成方式与文档导航 |
| [dataset/competition-dataset.md](dataset/competition-dataset.md) | **比赛官方字段表**：节点表和支路表的原始列名、含义与示例 |
| [dataset/competition-dataset-guide.md](dataset/competition-dataset-guide.md) | **数据集完整教程**：从 IEEE 39 算例到字段、掩码、打标和训练建议 |
| [dataset/model-design.md](dataset/model-design.md) | **神经网络模型设计**：v1 复盘、基线对比、输入/输出/损失/数据划分的 v2 方案（v1 已实现，v2 待实现） |
| [doc/case39/README.md](doc/case39/README.md) | IEEE 39 系统 10 章教程：数据、母线类型、牛顿法、结果解读 |
| [doc/powerflow-textbook/README.md](doc/powerflow-textbook/README.md) | 从数学、标幺值、Ybus 到牛顿法和代码实现的系统教材 |
| [doc/README.md](doc/README.md) | 文档总索引 |

建议阅读顺序：

1. `dataset/README.md`：先明确比赛任务、数据布局和字段约定。
2. `dataset/competition-dataset-guide.md`：系统理解数据从算例到样本的全过程。
3. `doc/case39/README.md`：把 39 节点系统的数据、母线类型、潮流结果弄懂。
4. `doc/powerflow-textbook/`：补理论，重点是 Ybus、功率方程、牛顿法。

## 运行 IEEE 39 教程

```bash
python src/examples/step3_case39.py
```

一个算例，两条求解路径互相验证：

| 路径 | 实现 | 迭代次数 |
|---|---|---:|
| A | PYPOWER `runpf` | 1（算例初值已接近收敛解） |
| B | 自研极坐标牛顿法 | 1；平启动 4 |

两个实现的最大电压差异：自研 vs PYPOWER 为 `2.2e-16` p.u.。

配套数据导出脚本：

```bash
python doc/case39/dump_case39_tables.py     # 原始数据表
python doc/case39/dump_case39_solution.py   # 潮流结果表
```

## 测试

```bash
python -m pytest -q
```

测试覆盖：

- case9 是否 4 次收敛；
- 自研极坐标雅可比是否与 PYPOWER 复数导数矩阵一致；
- 解析雅可比是否通过中心差分；
- 自研、PYPOWER 电压结果是否一致；
- 赛事数据集的拓扑、掩码、标签、能量平衡、落盘/读回一致性；
- GNN 的输入形状、归一化、已知电压复制、数据划分和一次训练冒烟。

只跑赛事数据集测试：

```bash
python -m pytest tests/test_competition_dataset.py -q
```

只跑 GNN 测试：

```bash
python -m pytest tests/test_model_gnn.py -q
```

## 环境清理

只删除 Python 虚拟环境：

```bash
rm -rf .venv
```

依赖和版本声明保留在 `requirements.txt` 与 `pyproject.toml` 中，可随时重建。
