"""实验 8：比较牛顿法、快速解耦和高斯-赛德尔潮流。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import prepare_case9  # noqa: E402
from pypower.api import fdpf, gausspf, makeB, newtonpf, ppoption  # noqa: E402


def main() -> None:
    problem = prepare_case9()
    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_TOL=1e-10,
        PF_MAX_IT=1000,
    )
    rows = []

    V, success, iterations = newtonpf(
        problem.Ybus,
        problem.Sbus,
        problem.V0,
        problem.ref,
        problem.pv,
        problem.pq,
        options,
    )
    rows.append(("NR", V, success, iterations))

    Bp, Bpp = makeB(problem.base_mva, problem.bus, problem.branch, 2)
    V, success, iterations = fdpf(
        problem.Ybus,
        problem.Sbus,
        problem.V0,
        Bp,
        Bpp,
        problem.ref,
        problem.pv,
        problem.pq,
        options,
    )
    rows.append(("FDXB", V, success, iterations))

    V, success, iterations = gausspf(
        problem.Ybus,
        problem.Sbus,
        problem.V0,
        problem.ref,
        problem.pv,
        problem.pq,
        options,
    )
    rows.append(("GS", V, success, iterations))

    print("alg    success   iterations   minVm(pu)    P-loss(MW)")
    print("-----  --------  -----------  ----------   ----------")
    for name, V, success, iterations in rows:
        losses = float(np.real(np.sum(V * np.conj(problem.Ybus.dot(V)))) * problem.base_mva)
        print(
            f"{name:5s}  {str(bool(success)):8s}  {iterations:11d}  "
            f"{float(np.min(np.abs(V))):10.6f}  {losses:10.6f}"
        )


if __name__ == "__main__":
    main()
