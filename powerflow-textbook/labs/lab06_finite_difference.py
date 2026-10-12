"""实验 6：用中心差分验证解析雅可比矩阵。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mismatch_from_x import mismatch_from_x  # noqa: E402
from powerflow import jacobian_polar, prepare_case9  # noqa: E402


def main() -> None:
    problem = prepare_case9()
    V0 = problem.V0
    Va0 = np.angle(V0)
    Vm0 = np.abs(V0)
    x0 = np.r_[Va0[problem.pv], Va0[problem.pq], Vm0[problem.pq]]

    J = jacobian_polar(problem.Ybus, V0, problem.pv, problem.pq)
    selected = [(0, 0), (0, 2), (3, 3), (7, 7), (8, 8), (8, 10), (9, 8), (13, 13)]
    step = 1e-6
    max_error = 0.0

    print("row  col      analytic      finite-diff      abs error")
    print("---  ---      --------      -----------      ---------")
    for row, col in selected:
        xp = x0.copy()
        xm = x0.copy()
        xp[col] += step
        xm[col] -= step
        Fp = mismatch_from_x(
            xp, problem.Ybus, problem.Sbus, problem.pv, problem.pq, Va0, Vm0
        )
        Fm = mismatch_from_x(
            xm, problem.Ybus, problem.Sbus, problem.pv, problem.pq, Va0, Vm0
        )
        derivative = (Fp[row] - Fm[row]) / (2.0 * step)
        error = abs(J[row, col] - derivative)
        max_error = max(max_error, error)
        print(
            f"{row + 1:3d}  {col + 1:3d}  {J[row, col]:12.8f}  "
            f"{derivative:13.8f}  {error:11.3e}"
        )

    print(f"\n最大绝对误差 = {max_error:.3e}")


if __name__ == "__main__":
    main()
