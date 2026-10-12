"""实验 4：手工完成并完整打印第一次牛顿迭代。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import solve_case9  # noqa: E402
from powerflow.report import print_step_details  # noqa: E402


def main() -> None:
    problem, result, _ = solve_case9(tol=1e-10)
    if not result.steps:
        raise RuntimeError("算例在迭代前已经收敛")
    print_step_details(result.steps[0], problem)
    print("\n第一次迭代完成。完整轨迹请运行 lab05_trace_newton.py。")


if __name__ == "__main__":
    main()
