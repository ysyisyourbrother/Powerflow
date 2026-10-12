# src：Python 算法与示例

```text
src/
├── powerflow/                  # 可复用 Python 包
│   ├── cases.py                # PYPOWER/Pandapower 算例准备和结果回算
│   ├── newton_raphson.py       # 公式显式展开的牛顿法求解器
│   ├── plotting.py             # 网络拓扑与电压图
│   └── report.py               # 教学表格输出
└── examples/
    ├── smoke_test.py           # 环境自检
    ├── step1_case9.py          # 自研牛顿法
    ├── step2_pandapower.py     # Pandapower 建模仿真
    ├── step3_case39.py         # IEEE 39 三路径完整教程
    └── compare_backends.py     # 三路径对照
```

## 运行

所有命令在项目根目录执行：

```bash
.venv/bin/python src/examples/step1_case9.py
.venv/bin/python src/examples/step2_pandapower.py
.venv/bin/python src/examples/step3_case39.py
.venv/bin/python src/examples/compare_backends.py
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
