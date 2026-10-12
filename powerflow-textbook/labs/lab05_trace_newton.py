"""实验 5：完整复现并对照牛顿法迭代轨迹。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import run_pypower_case9, solve_case9  # noqa: E402
from powerflow.report import mismatch_labels_for_problem, print_iteration_summary  # noqa: E402
from pypower.idx_bus import VA, VM  # noqa: E402


def main() -> None:
    problem, result, custom = solve_case9(tol=1e-10, max_iter=20)
    pypower, success = run_pypower_case9(tol=1e-10, max_iter=20)

    print("=" * 78)
    print("实验 5：牛顿法完整迭代轨迹")
    print("=" * 78)
    print_iteration_summary(result)

    labels = mismatch_labels_for_problem(problem)
    trace_path = Path(__file__).with_name("newton_trace.csv")
    with trace_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["iteration", "norm_before", "norm_after", "max_dx", *labels])
        writer.writerow([0, result.initial_norm, "", "", *result.initial_mismatch.tolist()])
        for step in result.steps:
            writer.writerow(
                [
                    step.iteration,
                    step.norm_before,
                    step.norm_after,
                    step.max_correction,
                    *step.mismatch_after.tolist(),
                ]
            )

    print("\n每步残差和修正量：")
    for step in result.steps:
        print(f"\n--- iteration {step.iteration} ---")
        print("dx =", np.array2string(step.correction, precision=10, suppress_small=False))
        print("F_after =", np.array2string(step.mismatch_after, precision=10, suppress_small=False))

    error_vm = np.max(np.abs(custom["bus"][:, VM] - pypower["bus"][:, VM]))
    error_va = np.max(np.abs(custom["bus"][:, VA] - pypower["bus"][:, VA]))
    print(f"\n与 PYPOWER 比较：success={success}")
    print(f"  最大电压幅值误差 = {error_vm:.3e} p.u.")
    print(f"  最大电压角度误差 = {error_va:.3e} deg")
    print(f"  完整轨迹已写入：{trace_path}")


if __name__ == "__main__":
    main()
