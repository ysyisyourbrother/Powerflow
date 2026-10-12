"""实验 1：建立 case9 潮流计算基线。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import solve_case9  # noqa: E402
from powerflow.report import print_iteration_summary, print_power_balance  # noqa: E402
from pypower.idx_bus import BUS_I, PD, VM  # noqa: E402
from pypower.idx_gen import PG  # noqa: E402


def main() -> None:
    _, result, external = solve_case9(tol=1e-10)
    total_gen = float(np.sum(external["gen"][:, PG]))
    total_load = float(np.sum(external["bus"][:, PD]))
    minimum_index = int(np.argmin(external["bus"][:, VM]))

    print("=" * 62)
    print("实验 1：case9 基础潮流")
    print("=" * 62)
    print_iteration_summary(result)
    print("\n关键摘要：")
    print(f"是否收敛      : {result.converged}")
    print(f"迭代次数      : {result.iterations}")
    print(f"总发电        : {total_gen:.4f} MW")
    print(f"总负荷        : {total_load:.4f} MW")
    print(f"网络有功损耗  : {total_gen - total_load:.4f} MW")
    print(f"最低电压母线  : {int(external['bus'][minimum_index, BUS_I])}")
    print(f"最低电压幅值  : {external['bus'][minimum_index, VM]:.6f} p.u.")
    print("\n功率平衡：")
    print_power_balance(external)


if __name__ == "__main__":
    main()
