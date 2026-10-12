"""教学脚本共用的表格打印函数。"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pypower.idx_bus import BUS_I, BUS_TYPE, PD, PQ, PV, QD, REF, VA, VM
from pypower.idx_gen import GEN_BUS, PG, QG

from .newton_raphson import NewtonResult, NewtonStep

BUS_TYPE_NAMES = {PQ: "PQ", PV: "PV", REF: "Slack"}


def format_vector(values: Iterable[float], precision: int = 8) -> str:
    return "[" + ", ".join(f"{value:.{precision}e}" for value in values) + "]"


def mismatch_labels(result: NewtonResult, problem) -> list[str]:
    p_labels = [f"ΔP{bus + 1}" for bus in np.r_[problem.pv, problem.pq]]
    q_labels = [f"ΔQ{bus + 1}" for bus in problem.pq]
    return p_labels + q_labels


def print_iteration_summary(result: NewtonResult) -> None:
    print("iteration   ||F_before||inf   ||dx||inf   ||F_after||inf")
    print("---------   ---------------   ---------   --------------")
    print(f"{0:9d}   {result.initial_norm:15.8e}   {'-':>9}   {'-':>14}")
    for step in result.steps:
        print(
            f"{step.iteration:9d}   {step.norm_before:15.8e}   "
            f"{step.max_correction:9.3e}   {step.norm_after:14.8e}"
        )


def print_step_details(step: NewtonStep, problem) -> None:
    labels = mismatch_labels_for_problem(problem)
    print(f"\n===== 第 {step.iteration} 次迭代 =====")
    print("迭代前失配 F:")
    for label, value in zip(labels, step.mismatch_before):
        print(f"  {label:>5}: {value: .12e}")
    print(f"||F||inf = {step.norm_before:.12e}")
    print("\nJ(x) =")
    np.set_printoptions(precision=6, suppress=True, linewidth=180)
    print(step.jacobian)
    print("\ndx = J^{-1}(-F):")
    print(step.correction)
    print(f"||dx||inf = {step.max_correction:.12e}")
    print("\n迭代后复电压 V = |V|∠θ:")
    for bus, voltage in enumerate(step.voltage_after, start=1):
        print(f"  bus {bus}: {abs(voltage):.12f} ∠ {np.angle(voltage, deg=True): .9f}°")
    print(f"\n迭代后 ||F||inf = {step.norm_after:.12e}")


def mismatch_labels_for_problem(problem) -> list[str]:
    p_labels = [f"ΔP{bus + 1}" for bus in np.r_[problem.pv, problem.pq]]
    q_labels = [f"ΔQ{bus + 1}" for bus in problem.pq]
    return p_labels + q_labels


def print_bus_results(result: dict) -> None:
    print("bus   type      Vm(p.u.)    Va(deg)    load P(MW)  load Q(Mvar)")
    print("---   ----      --------    -------       -----      --------")
    for row in result["bus"]:
        bus_number = int(row[BUS_I])
        bus_type = BUS_TYPE_NAMES.get(int(row[BUS_TYPE]), "?")
        print(
            f"{bus_number:3d}   {bus_type:4s}   {row[VM]:10.6f}  {row[VA]:10.5f}  "
            f"{row[PD]:11.5f}  {row[QD]:12.5f}"
        )


def print_generator_results(result: dict) -> None:
    print("gen   bus       P(MW)      Q(Mvar)")
    print("---   ---       -----      --------")
    for index, row in enumerate(result["gen"], start=1):
        print(
            f"{index:3d}   {int(row[GEN_BUS]):3d}   {row[PG]:10.5f}  "
            f"{row[QG]:11.5f}"
        )


def print_power_balance(result: dict) -> None:
    total_gen = float(np.sum(result["gen"][:, PG]))
    total_load = float(np.sum(result["bus"][:, PD]))
    loss = total_gen - total_load
    print(f"总发电 = {total_gen:.6f} MW")
    print(f"总负荷 = {total_load:.6f} MW")
    print(f"网络有功损耗 = {loss:.6f} MW")


def print_voltage_comparison(
    custom: dict,
    pypower: dict,
    pandapower_net,
) -> None:
    print("bus   custom Vm   PYPOWER Vm   Pandapower Vm   max|ΔVm|")
    print("---   ---------   -----------   --------------   ---------")
    pp_vm = pypower["bus"][:, VM]
    pandapower_vm = pandapower_net.res_bus["vm_pu"].to_numpy()
    for index, row in enumerate(custom["bus"]):
        number = int(row[BUS_I])
        difference = max(abs(row[VM] - pp_vm[index]), abs(row[VM] - pandapower_vm[index]))
        print(
            f"{number:3d}   {row[VM]:9.6f}   {pp_vm[index]:11.6f}   "
            f"{pandapower_vm[index]:14.6f}   {difference:9.2e}"
        )
