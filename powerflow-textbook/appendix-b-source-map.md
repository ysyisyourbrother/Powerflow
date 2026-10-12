# 附录 B：源码地图

## B.1 总调用链

```text
src/examples/step1_case9.py
        │
        ▼
powerflow.solve_case9()
        │
        ├── prepare_case9()
        │     ├── pypower.case9()
        │     ├── pypower.ext2int()
        │     ├── pypower.makeYbus()
        │     ├── pypower.makeSbus()
        │     └── pypower.bustypes()
        │
        ├── solve_newton_raphson()
        │     ├── polar_power_equations()
        │     ├── mismatch_vector()
        │     ├── jacobian_polar()
        │     ├── np.linalg.solve()
        │     └── 状态更新
        │
        └── build_external_result()
              ├── pypower.pfsoln()
              └── pypower.int2ext()
```

## B.2 `src/powerflow/newton_raphson.py`

| 对象 | 作用 |
|---|---|
| `NewtonStep` | 保存一次迭代的 F、J、dx、V、更新后 F |
| `NewtonResult` | 保存最终电压、收敛状态和全部迭代 |
| `polar_power_equations` | 计算所有母线 P、Q |
| `mismatch_vector` | 构造 14 维残差 |
| `jacobian_polar` | 显式构造四块雅可比 |
| `solve_newton_raphson` | 牛顿法主循环 |

主循环顺序：

```text
F
→ J
→ solve(J, -F)
→ update Va/Vm
→ rebuild V
→ new F
→ convergence check
```

## B.3 `src/powerflow/cases.py`

| 对象 | 作用 |
|---|---|
| `PowerFlowProblem` | 封装 `Ybus/Sbus/ref/pv/pq/V0` |
| `_pad_branch_columns` | 补齐线路潮流输出列 |
| `_initial_voltage` | 构造复电压初值 |
| `prepare_case9` | 完成 case9 准备 |
| `build_external_result` | 回算并恢复外部编号 |
| `solve_case9` | 一步完成准备、求解、回算 |
| `run_pypower_case9` | PYPOWER 参考求解 |
| `run_pandapower_case9` | Pandapower 工程求解 |

## B.4 `src/powerflow/report.py`

| 函数 | 输出 |
|---|---|
| `print_iteration_summary` | 迭代范数表 |
| `print_step_details` | 单次迭代的 F、J、dx、V |
| `print_bus_results` | 母线电压和负荷 |
| `print_generator_results` | 发电机 P、Q |
| `print_power_balance` | 发电、负荷、损耗 |
| `print_voltage_comparison` | 三个求解路径电压对照 |

## B.5 `src/powerflow/plotting.py`

| 函数 | 输出 |
|---|---|
| `_layout` | 根据支路图生成节点布局 |
| `plot_power_flow` | 线路负载率、母线类型、电压柱状图 |

## B.6 示例脚本

| 文件 | 作用 |
|---|---|
| `smoke_test.py` | 运行自研、PYPOWER、Pandapower 并断言一致 |
| `step1_case9.py` | 打印自研迭代并绘图 |
| `step2_pandapower.py` | 展示 Pandapower 元件表和结果表 |
| `compare_backends.py` | 比较三个求解路径 |

## B.7 实验脚本

| 文件 | 内容 |
|---|---|
| `lab01_run_case9.py` | 基线结果 |
| `lab02_inspect_case9.py` | 数据表和母线分类 |
| `lab03_build_matrices.py` | `Ybus/Sbus/V0` |
| `lab04_first_iteration.py` | 第一次完整迭代 |
| `lab05_trace_newton.py` | 四次完整迭代和 CSV |
| `lab06_finite_difference.py` | 解析/数值雅可比 |
| `lab07_perturb_load.py` | 负荷扰动 |
| `lab08_compare_algorithms.py` | NR/FDXB/GS 比较 |

## B.8 PYPOWER 内部函数

| 函数 | 本项目中的作用 |
|---|---|
| `case9` | 原始数据 |
| `ext2int` | 0 基连续编号 |
| `makeYbus` | `Ybus/Yf/Yt` |
| `makeSbus` | 计划注入 |
| `bustypes` | 母线分类 |
| `dSbus_dV` | 复数导数参考 |
| `newtonpf` | 成熟 NR |
| `fdpf` | 快速解耦 |
| `gausspf` | Gauss-Seidel |
| `pfsoln` | 结果回算 |
| `int2ext` | 恢复外部编号 |

## B.9 Pandapower 内部接口

| 接口 | 本项目中的作用 |
|---|---|
| `from_ppc` | PYPOWER 字典转 Pandapower 网络 |
| `runpp` | 工程化潮流求解 |
| `net.res_bus` | 母线结果 |
| `net.res_gen` | 发电机结果 |
| `net.res_ext_grid` | 平衡发电机结果 |
| `net.res_line` | 线路结果 |

## B.10 测试

| 文件 | 验证 |
|---|---|
| `tests/test_newton_raphson.py` | 收敛、雅可比、库对照 |
| `tests/test_finite_difference.py` | 中心差分 |

运行：

```bash
.venv/bin/python -m pytest
```

## B.11 扩展时应放在哪里

- 新算法：`src/powerflow/`
- 新算例适配：`src/powerflow/cases.py` 或新模块
- 新实验：`src/examples/` 或 `powerflow-textbook/labs/`
- 新测试：`tests/`
- 比赛输入：`dataset/`
