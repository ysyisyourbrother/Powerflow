# src：Python 算法与示例

```text
src/
├── powerflow/                  # 可复用 Python 包
│   ├── cases.py                # PYPOWER 算例准备和结果回算
│   ├── newton_raphson.py       # 公式显式展开的牛顿法求解器
│   ├── plotting.py             # 网络拓扑与电压图
│   ├── report.py               # 教学表格输出
│   ├── dataset/
│   │   ├── competition.py      # 赛事格式：schema / 掩码 / 打标 / 读写
│   │   └── sampler.py          # 负荷扰动采样
│   └── model/
│       ├── data.py             # 归一化、图张量、数据划分
│       ├── gnn.py              # 边感知 GNN
│       └── training.py         # 损失、训练循环、评估
└── examples/
    ├── smoke_test.py           # 环境自检
    ├── step1_case9.py          # 自研牛顿法
    ├── step3_case39.py         # IEEE 39 两条路径完整教程
    ├── compare_backends.py     # 两条路径对照
    ├── build_competition_dataset.py # 赛事对齐数据集生成（当前主用）
    ├── train_gnn.py            # 训练 GNN
    └── evaluate_gnn.py         # 重新评估 checkpoint
```

## 运行

所有命令在项目根目录执行。如果使用 `install.sh` 创建的虚拟环境，把
`python` 换成 `.venv/bin/python`：

```bash
python src/examples/step1_case9.py
python src/examples/step3_case39.py
python src/examples/compare_backends.py
python src/examples/build_competition_dataset.py --n-samples 2000
python src/examples/train_gnn.py --device cuda --epochs 150
python src/examples/evaluate_gnn.py --checkpoint runs/gnn_v1/best.pt --device cuda
```

## 最重要的代码入口

```python
from powerflow import prepare_case9, solve_newton_raphson, build_external_result

problem = prepare_case9()
result = solve_newton_raphson(problem, tol=1e-10, max_iter=20)
external = build_external_result(problem, result)
```

- `problem.Ybus`：复数节点导纳矩阵；
- `problem.Sbus`：计划复功率注入；
- `problem.pv`、`problem.pq`：母线分类；
- `result.steps`：每次迭代的 `F`、`J`、`dx` 和更新后电压；
- `external`：外部编号下的母线、发电机和线路结果。

新增实验代码放在 `src/examples/`。可复用的算法函数放在 `src/powerflow/`。
