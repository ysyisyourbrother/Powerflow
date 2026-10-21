"""算例准备、参考求解器运行和结果回算。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
from pypower.api import (
    bustypes,
    case9,
    case39,
    ext2int,
    int2ext,
    makeSbus,
    makeYbus,
    pfsoln,
    ppoption,
    runpf,
)
from pypower.idx_brch import QT
from pypower.idx_bus import BUS_I, BUS_TYPE, PQ, VA, VM
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


def pad_branch_columns(ppc: dict[str, Any]) -> dict[str, Any]:
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


def prepare_case(ppc: dict[str, Any]) -> PowerFlowProblem:
    """把任意 PYPOWER 算例转换到内部连续编号，并建立完整数据结构。"""

    ppc = pad_branch_columns(ppc)
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


def external_bus_rows(problem: PowerFlowProblem) -> np.ndarray:
    """内部行号 → ``problem.ppc["bus"]`` 的行号。

    ``ext2int`` 会把母线重排成连续编号，所以 ``ppci["bus"]`` 的第 i 行
    并不对应 ``ppc["bus"]`` 的第 i 行。要同时更新两份数据时，
    必须先过这层映射。

    ``ppci["order"]["bus"]["i2e"][i]`` 给出内部行 i 的**外部母线编号**，
    再用编号去 ``ppc["bus"]`` 里查行。按编号查而不是按位置算，
    是为了不依赖"外部编号恰好是 1..n 连续"这个巧合。
    """

    external_ids = problem.ppc["bus"][:, BUS_I].astype(int)
    row_of_number = {int(number): row for row, number in enumerate(external_ids.tolist())}
    i2e = np.asarray(problem.ppci["order"]["bus"]["i2e"], dtype=int)
    return np.asarray([row_of_number[int(number)] for number in i2e], dtype=int)


def flat_start(ppc: dict[str, Any]) -> dict[str, Any]:
    """深拷贝 ``ppc``，并把 bus 的 VM/VA 抹平成 ``1.0 / 0.0``。

    PYPOWER 的许多算例（例如 ``case39``）自带的 ``VM``/``VA`` 本身就是一组
    已求解的潮流结果。若直接拿它当启动点，牛顿法一次迭代就收敛，"迭代次数"
    这类元数据会失去意义。做数据生成时必须先抹掉。
    """

    flat = copy.deepcopy(ppc)
    flat["bus"] = np.asarray(flat["bus"], dtype=float)
    flat["bus"][:, VM] = 1.0
    flat["bus"][:, VA] = 0.0
    return flat


def prepare_case9() -> PowerFlowProblem:
    """读取 PYPOWER 自带的 case9，并建立完整内部数据结构。"""

    return prepare_case(case9())


def prepare_case39(*, flat: bool = True) -> PowerFlowProblem:
    """读取 PYPOWER 自带的 case39，并建立完整内部数据结构。

    ``flat=True``（默认）会先抹掉算例自带的那组解，改用平启动。
    此时内部初值为：PQ 母线 ``|V0| = 1.0``、PV 与平衡母线 ``|V0| = VG``、
    所有母线相角为 0。
    """

    ppc = case39()
    return prepare_case(flat_start(ppc) if flat else ppc)


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


def run_pypower_pf(
    ppc: dict[str, Any],
    tol: float = 1e-10,
    max_iter: int = 40,
) -> tuple[dict[str, Any], bool]:
    """用 PYPOWER 自带牛顿法求解任意 ``ppc``，作为独立参照实现。"""

    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=tol, PF_MAX_IT=max_iter)
    return runpf(pad_branch_columns(ppc), options)


def run_pypower_case9(
    tol: float = 1e-10,
    max_iter: int = 20,
) -> tuple[dict[str, Any], bool]:
    """使用 PYPOWER 自带牛顿法求解，作为教学代码的参考实现。"""

    return run_pypower_pf(case9(), tol=tol, max_iter=max_iter)
