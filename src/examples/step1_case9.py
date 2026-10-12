"""入门示例：用自研 Python 牛顿法求解 IEEE 9 节点系统。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow import solve_case9  # noqa: E402
from powerflow.plotting import plot_power_flow  # noqa: E402
from powerflow.report import (  # noqa: E402
    print_bus_results,
    print_generator_results,
    print_iteration_summary,
    print_power_balance,
)
from pypower.idx_bus import BUS_TYPE, PD, PQ, PV, REF  # noqa: E402


def main() -> None:
    problem, result, external = solve_case9(tol=1e-10)
    bus = problem.ppc["bus"]

    print("=" * 62)
    print("潮流计算入门 1：IEEE 9 节点系统（PYPOWER 数据 + Python 求解器）")
    print("=" * 62)
    print(f"母线数: {len(bus)}，发电机数: {len(problem.ppc['gen'])}，"
          f"线路数: {len(problem.ppc['branch'])}")
    for bus_type, name in ((PQ, "PQ 负荷节点"), (PV, "PV 发电节点"), (REF, "平衡节点")):
        count = int(np.sum(bus[:, BUS_TYPE].astype(int) == bus_type))
        print(f"  type {bus_type} = {name}: {count} 个")
    print(f"总负荷: {np.sum(bus[:, PD]):.2f} MW")

    print("\n牛顿-拉夫逊迭代摘要")
    print_iteration_summary(result)

    print("\n母线结果")
    print_bus_results(external)
    print("\n发电机结果")
    print_generator_results(external)
    print("\n功率平衡")
    print_power_balance(external)

    outfile = Path(__file__).with_name("case9_result.png")
    saved = plot_power_flow(external, outfile, title="IEEE 9-bus: custom Newton-Raphson")
    print(f"\n图已保存: {saved}")


if __name__ == "__main__":
    main()
