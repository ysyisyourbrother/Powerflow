"""算例准备、参考求解器运行和结果回算。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from pandapower.converter.pypower.from_ppc import from_ppc
from pypower.api import (
    bustypes,
    case9,
    ext2int,
    int2ext,
    makeSbus,
    makeYbus,
    pfsoln,
    ppoption,
    runpf,
)
from pypower.idx_brch import QT
from pypower.idx_bus import BUS_TYPE, PQ, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG

from .newton_raphson import NewtonResult, solve_newton_raphson


@dataclass(frozen=True)
class PowerFlowProblem:
    """已经转换到内部连续编号的 PYPOWER 潮流问题。"""

    base_mva: float
    ppc: dict[str, Any]
    ppci: dict[str, Any]
    Ybus: Any
    Yf: Any
    Yt: Any
    Sbus: np.ndarray
    ref: np.ndarray
    pv: np.ndarray
    pq: np.ndarray
    V0: np.ndarray

    @property
    def bus(self) -> np.ndarray:
        return self.ppci["bus"]

    @property
    def gen(self) -> np.ndarray:
        return self.ppci["gen"]

    @property
    def branch(self) -> np.ndarray:
        return self.ppci["branch"]

    @property
    def nb(self) -> int:
        return self.bus.shape[0]


def _pad_branch_columns(ppc: dict[str, Any]) -> dict[str, Any]:
    """补齐 PF/QF/PT/QT 列，使输入也能直接交给 pfsoln。"""

    ppc = copy.deepcopy(ppc)
    ppc["bus"] = np.asarray(ppc["bus"], dtype=float)
    ppc["gen"] = np.asarray(ppc["gen"], dtype=float)
    ppc["branch"] = np.asarray(ppc["branch"], dtype=float)
    branch = ppc["branch"]
    if branch.shape[1] <= QT:
        ppc["branch"] = np.c_[
            branch,
            np.zeros((branch.shape[0], QT - branch.shape[1] + 1)),
        ]
    return ppc


def _initial_voltage(ppci: dict[str, Any], pq: np.ndarray) -> np.ndarray:
    """由 bus 的 VM/VA 初值构造复电压，并恢复非 PQ 母线的发电机电压设定。"""

    bus = ppci["bus"]
    gen = ppci["gen"]
    V0 = bus[:, VM] * np.exp(1j * np.deg2rad(bus[:, VA]))

    on = np.flatnonzero(gen[:, GEN_STATUS] > 0)
    gbus = gen[on, GEN_BUS].astype(int)
    controlled = bus[gbus, BUS_TYPE] != PQ
    for bus_index, gen_index in zip(gbus[controlled], on[controlled]):
        magnitude = gen[gen_index, VG]
        V0[bus_index] = magnitude * np.exp(1j * np.angle(V0[bus_index]))
    return V0.astype(complex)


def prepare_case9() -> PowerFlowProblem:
    """读取 PYPOWER 自带的 case9，并建立完整内部数据结构。"""

    ppc = _pad_branch_columns(case9())
    ppci = ext2int(copy.deepcopy(ppc))
    base_mva = float(ppci["baseMVA"])
    Ybus, Yf, Yt = makeYbus(base_mva, ppci["bus"], ppci["branch"])
    Sbus = makeSbus(base_mva, ppci["bus"], ppci["gen"])
    ref, pv, pq = bustypes(ppci["bus"], ppci["gen"])
    V0 = _initial_voltage(ppci, pq)
    return PowerFlowProblem(
        base_mva=base_mva,
        ppc=ppc,
        ppci=ppci,
        Ybus=Ybus,
        Yf=Yf,
        Yt=Yt,
        Sbus=Sbus,
        ref=np.asarray(ref, dtype=int),
        pv=np.asarray(pv, dtype=int),
        pq=np.asarray(pq, dtype=int),
        V0=V0,
    )


def build_external_result(
    problem: PowerFlowProblem,
    result: NewtonResult,
) -> dict[str, Any]:
    """由最终复电压回算发电机出力和线路潮流，并转回外部编号。"""

    bus = problem.ppci["bus"].copy()
    gen = problem.ppci["gen"].copy()
    branch = problem.ppci["branch"].copy()

    bus, gen, branch = pfsoln(
        problem.base_mva,
        bus,
        gen,
        branch,
        problem.Ybus,
        problem.Yf,
        problem.Yt,
        result.voltage,
        problem.ref,
        problem.pv,
        problem.pq,
    )

    solved_internal = dict(problem.ppci)
    solved_internal.update(
        {
            "bus": bus,
            "gen": gen,
            "branch": branch,
            "success": int(result.converged),
            "iterations": result.iterations,
        }
    )
    solved_external = int2ext(solved_internal)
    solved_external["iterations"] = result.iterations
    return solved_external


def solve_case9(
    tol: float = 1e-10,
    max_iter: int = 20,
) -> tuple[PowerFlowProblem, NewtonResult, dict[str, Any]]:
    """一次性完成 case9 的建模、自研求解和结果回算。"""

    problem = prepare_case9()
    result = solve_newton_raphson(problem, tol=tol, max_iter=max_iter)
    external = build_external_result(problem, result)
    return problem, result, external


def run_pypower_case9(
    tol: float = 1e-10,
    max_iter: int = 20,
) -> tuple[dict[str, Any], bool]:
    """使用 PYPOWER 自带牛顿法求解，作为教学代码的参考实现。"""

    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=tol, PF_MAX_IT=max_iter)
    return runpf(_pad_branch_columns(case9()), options)


def run_pandapower_case9(
    tol_mva: float = 1e-10,
    max_iter: int = 20,
):
    """把同一个 PYPOWER case9 转成 Pandapower 网络并运行牛顿法。

    使用同一个源算例可以排除“两个库自带 case9 编号不同”造成的干扰。
    """

    import pandapower as pp

    problem = prepare_case9()
    net = from_ppc(copy.deepcopy(problem.ppc), f_hz=60)
    pp.runpp(
        net,
        algorithm="nr",
        calculate_voltage_angles=True,
        init="flat",
        tolerance_mva=tol_mva,
        max_iteration=max_iter,
        numba=False,
    )
    return net
