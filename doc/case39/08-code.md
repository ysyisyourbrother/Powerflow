# 第四部分　完整代码与运行

> 配套脚本、运行命令、脚本结构、最小可运行示例

## 4.1 配套脚本

| 文件 | 作用 |
|---|---|
| [src/examples/step3_case39.py](../../src/examples/step3_case39.py) | 本教程的完整可运行示例（六步） |
| [dump_case39_tables.py](dump_case39_tables.py) | 打印本教程第一部分的所有数据表 |
| [dump_case39_solution.py](dump_case39_solution.py) | 打印本教程第三部分的所有结果表 |
| [src/examples/case9_result.png](../../src/examples/case9_result.png) | case9 的网络图（对比参考） |

## 4.2 运行

在仓库根目录运行：

```bash
# 完整教程示例（约 10 秒）
MPLCONFIGDIR=/tmp/mpl_cache .venv/bin/python src/examples/step3_case39.py

# 原始数据表（脚本会自动定位 src/，也可从其他目录运行）
MPLCONFIGDIR=/tmp/mpl_cache .venv/bin/python doc/case39/dump_case39_tables.py

# 结果表
MPLCONFIGDIR=/tmp/mpl_cache .venv/bin/python doc/case39/dump_case39_solution.py
```

如果使用已激活的 conda 环境，把 `.venv/bin/python` 换成 `python` 即可。

## 4.3 `step3_case39.py` 的结构

脚本按 `第 N 步` 打印分隔标题，建议对照本教程逐段阅读：

| 步骤 | 函数 | 对应章节 |
|---|---|---|
| 1 | `show_case_data()` | 第一部分：数据总览 |
| 2 | `solve_with_pypower()` | 2.8 节：PYPOWER 求解 |
| 3 | `solve_with_custom_newton()` | 2.5 ~ 2.7 节：牛顿法原理 |
| 3b | `flat_start_experiment()` | 2.10 节：平启动与二次收敛 |
| 4 | `show_results()` | 第三部分：结果解读 |
| 5 | `reactive_balance()` | 3.4 节：无功平衡拆解 |
| 6 | `compare_backends()` | 2.11 节：两路交叉验证 |

脚本里有三个可复用的工具函数，值得单独理解：

```python
# 补齐 branch 表的 PF/QF/PT/QT 列，保证能直接交给 pfsoln 回算
pad_branch_columns(ppc)

# 把任意 PYPOWER 算例转成自研求解器需要的数据结构
prepare_case(ppc)

# 注意：原项目的 prepare_case9() 是 case9 专用的，
# 这里的 prepare_case(ppc) 做了泛化，可以处理 case39 等任何算例
```

## 4.4 最小可运行示例

如果你只想记住一段最短的代码，就是这 6 行：

```python
from pypower.api import case39, runpf, ppoption

ppc = case39()
result, success = runpf(
    ppc,
    ppoption(VERBOSE=1, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-10, PF_MAX_IT=20),
)

print("收敛:", success)
print("母线电压范围:", result["bus"][:, 7].min(), "~", result["bus"][:, 7].max(), "pu")
print("总有功出力:", result["gen"][:, 1].sum(), "MW")
print("总有功损耗:", result["gen"][:, 1].sum() - result["bus"][:, 2].sum(), "MW")
```


---

[← 结果解读：支路潮流](07-results-flow.md)　|　[返回目录](README.md)　|　[常见问题与排错 →](09-faq.md)
