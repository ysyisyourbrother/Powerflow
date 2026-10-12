"""实验 2：检查 case9 的数据模型和母线分类。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from powerflow import prepare_case9  # noqa: E402
from pypower.idx_brch import BR_B, BR_R, BR_STATUS, BR_X, F_BUS, T_BUS  # noqa: E402
from pypower.idx_bus import BUS_I, BUS_TYPE, PD, QD, VA, VM  # noqa: E402
from pypower.idx_gen import GEN_BUS, GEN_STATUS, PG, QG, VG  # noqa: E402


def main() -> None:
    problem = prepare_case9()
    bus = problem.ppc["bus"]
    gen = problem.ppc["gen"]
    branch = problem.ppc["branch"]

    print("规模：")
    print(f"  母线数量   : {len(bus)}")
    print(f"  发电机数量 : {len(gen)}")
    print(f"  支路数量   : {len(branch)}")
    print(f"  baseMVA    : {problem.base_mva:.1f} MVA")
    print(f"\nref = {problem.ref + 1}")
    print(f"pv  = {problem.pv + 1}")
    print(f"pq  = {problem.pq + 1}")
    print(f"PQ 数量={len(problem.pq)}, PV 数量={len(problem.pv)}, "
          f"平衡数量={len(problem.ref)}")

    print("\n母线主要数据：")
    print("bus  type    Pd(MW)  Qd(Mvar)    Vm0     Va0(deg)")
    for row in bus:
        print(
            f"{int(row[BUS_I]):3d}  {int(row[BUS_TYPE]):4d}  "
            f"{row[PD]:8.2f}  {row[QD]:9.2f}  {row[VM]:7.4f}  {row[VA]:9.2f}"
        )

    print("\n发电机主要数据：")
    print("bus   Pg(MW)  Qg(Mvar)    Vg    status")
    for row in gen:
        print(
            f"{int(row[GEN_BUS]):3d}  {row[PG]:7.2f}  {row[QG]:8.2f}  "
            f"{row[VG]:6.4f}  {int(row[GEN_STATUS]):6d}"
        )

    print("\n支路主要数据：")
    print("from  to       r        x        b   status")
    for row in branch:
        print(
            f"{int(row[F_BUS]):4d} {int(row[T_BUS]):3d}  {row[BR_R]:8.4f}  "
            f"{row[BR_X]:8.4f}  {row[BR_B]:8.4f}  {int(row[BR_STATUS]):6d}"
        )

    print(f"\n总负荷     : {bus[:, PD].sum():.2f} MW + j{bus[:, QD].sum():.2f} Mvar")
    print(f"发电初始值 : {gen[:, PG].sum():.2f} MW")


if __name__ == "__main__":
    main()
