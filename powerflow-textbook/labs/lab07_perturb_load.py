"""实验 7：改变母线 9 有功负荷，观察电压、发电和损耗。"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import build_external_result, prepare_case9, solve_newton_raphson  # noqa: E402
from pypower.idx_bus import PD, VM  # noqa: E402
from pypower.idx_gen import PG  # noqa: E402


def main() -> None:
    problem = prepare_case9()
    pd_values = [125.0, 150.0, 175.0, 200.0, 225.0]
    original_pd_9 = 125.0
    bus_9_index = 8

    print("Pd9(MW)  converged  iter        Vm9(pu)      Gen(MW)     Loss(MW)")
    print("-------  ---------  ----        --------     --------    --------")
    for pd_9 in pd_values:
        sbus = problem.Sbus.copy()
        sbus[bus_9_index] += -(pd_9 - original_pd_9) / problem.base_mva
        modified = replace(problem, Sbus=sbus)
        result = solve_newton_raphson(modified, tol=1e-10, max_iter=30)
        external = build_external_result(modified, result)
        total_gen = float(np.sum(external["gen"][:, PG]))
        total_load = float(np.sum(external["bus"][:, PD]))
        print(
            f"{pd_9:7.1f}  {str(result.converged):9s}  {result.iterations:4d}  "
            f"{external['bus'][bus_9_index, VM]:12.6f}  {total_gen:12.4f}  "
            f"{total_gen - total_load:11.4f}"
        )


if __name__ == "__main__":
    main()
