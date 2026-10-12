"""实验 3：构造 Ybus、V0、Sbus，并手算支路 4-5。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import prepare_case9  # noqa: E402
from pypower.idx_brch import BR_B, BR_R, BR_X, F_BUS, T_BUS  # noqa: E402


def main() -> None:
    problem = prepare_case9()
    branch = problem.branch
    # PYPOWER 内部使用 0 基索引：外部母线 4、5 对应内部索引 3、4。
    matches = np.flatnonzero(
        (branch[:, F_BUS].astype(int) == 3) & (branch[:, T_BUS].astype(int) == 4)
    )
    if not len(matches):
        raise RuntimeError("没有找到支路 4-5")

    row = branch[matches[0]]
    r = float(row[BR_R])
    x = float(row[BR_X])
    b = float(row[BR_B])
    y = 1.0 / (r + 1j * x)
    yff = y + 1j * b / 2.0
    yft = -y

    print("支路 4-5 手算：")
    print(f"  r = {r:.6f}, x = {x:.6f}, b = {b:.6f}")
    print(f"  y   = {y.real:.6f} + j{y.imag:.6f}")
    print(f"  Yff = {yff.real:.6f} + j{yff.imag:.6f}")
    print(f"  Yft = {yft.real:.6f} + j{yft.imag:.6f}")

    print("\n初始复电压：")
    print("bus       Re(V)       Im(V)          |V|    angle(deg)")
    for index, V in enumerate(problem.V0, start=1):
        print(
            f"{index:3d}  {V.real:11.6f}  {V.imag:11.6f}  "
            f"{abs(V):11.6f}  {np.angle(V, deg=True):11.6f}"
        )

    print("\n计划功率注入 Sbus = generation - load：")
    print("bus        Re(Sbus)       Im(Sbus)")
    for index, value in enumerate(problem.Sbus, start=1):
        print(f"{index:3d}  {value.real:13.8f}  {value.imag:13.8f}")

    Y = problem.Ybus.toarray()
    print("\nYbus 非零元素（只显示上三角和对角，避免重复）：")
    print("row  col          real            imag")
    for i in range(Y.shape[0]):
        for j in range(i, Y.shape[1]):
            if abs(Y[i, j]) > 1e-14:
                print(f"{i + 1:3d}  {j + 1:3d}  {Y[i, j].real:14.8f}  {Y[i, j].imag:14.8f}")

    asymmetry = np.linalg.norm(Y - Y.T, ord=np.inf)
    print(f"\nYbus 对称性误差 ||Y-Y^T||inf = {asymmetry:.3e}")


if __name__ == "__main__":
    main()
