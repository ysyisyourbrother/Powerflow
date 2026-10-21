"""一个透明、可逐步打印的极坐标牛顿-拉夫逊潮流求解器。

本模块刻意把 P/Q 方程和雅可比四块展开写成公式形式，便于与教材逐行对照。
生产环境可直接使用 ``pypower.newtonpf``；本模块的作用是把
黑盒打开，让每个中间量都可以观察和验证。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查，避免循环导入
    from .cases import PowerFlowProblem


@dataclass
class NewtonStep:
    """保存一次完整的牛顿迭代，供讲义和实验使用。"""

    iteration: int
    mismatch_before: np.ndarray
    norm_before: float
    P: np.ndarray
    Q: np.ndarray
    jacobian: np.ndarray
    correction: np.ndarray
    max_correction: float
    voltage_after: np.ndarray
    mismatch_after: np.ndarray
    norm_after: float


@dataclass
class NewtonResult:
    """牛顿潮流结果及完整迭代轨迹。"""

    voltage: np.ndarray
    converged: bool
    iterations: int
    initial_mismatch: np.ndarray
    initial_norm: float
    final_mismatch: np.ndarray
    final_norm: float
    steps: list[NewtonStep] = field(default_factory=list)


def _dense(matrix: Any) -> np.ndarray:
    """把 scipy 稀疏矩阵或普通数组统一转换为稠密复数数组。"""

    if hasattr(matrix, "toarray"):
        return np.asarray(matrix.toarray(), dtype=complex)
    return np.asarray(matrix, dtype=complex)


def polar_power_equations(Ybus: Any, V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r"""按极坐标公式计算所有母线的 P、Q。

    .. math::

        P_i=V_i\sum_k V_k(G_{ik}\cos\theta_{ik}+B_{ik}\sin\theta_{ik})

        Q_i=V_i\sum_k V_k(G_{ik}\sin\theta_{ik}-B_{ik}\cos\theta_{ik})
    """

    Y = _dense(Ybus)
    G = Y.real
    B = Y.imag
    Vm = np.abs(V)
    Va = np.angle(V)
    theta = Va[:, None] - Va[None, :]

    P = Vm * np.sum(Vm[None, :] * (G * np.cos(theta) + B * np.sin(theta)), axis=1)
    Q = Vm * np.sum(Vm[None, :] * (G * np.sin(theta) - B * np.cos(theta)), axis=1)
    return P, Q


def mismatch_vector(
    Ybus: Any,
    Sbus: np.ndarray,
    V: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
) -> np.ndarray:
    r"""构造真实数残差向量 ``F = [ΔP_PV, ΔP_PQ, ΔQ_PQ]``。

    复功率注入为

    .. math::

        S_{\mathrm{calc}}(V)=V\circ\operatorname{conj}(Y_{\mathrm{bus}}V)

    残差为

    .. math::

        \Delta S=S_{\mathrm{calc}}-S_{\mathrm{spec}}.
    """

    V = np.asarray(V, dtype=complex)
    S_calc = V * np.conj(Ybus.dot(V))
    delta_S = S_calc - Sbus
    return np.r_[delta_S[pv].real, delta_S[pq].real, delta_S[pq].imag]


def jacobian_polar(
    Ybus: Any,
    V: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
) -> np.ndarray:
    r"""显式构造极坐标潮流雅可比矩阵。

    返回矩阵按以下顺序排列：

    .. math::

        J=
        \begin{bmatrix}
        \partial P/\partial\theta & \partial P/\partial V_m\\
        \partial Q/\partial\theta & \partial Q/\partial V_m
        \end{bmatrix}.

    其中 P 方程对应 ``pv`` 后接 ``pq``，Q 方程只对应 ``pq``，状态变量也按
    ``[theta_pv, theta_pq, Vm_pq]`` 排列。
    """

    Y = _dense(Ybus)
    G = Y.real
    B = Y.imag
    Vm = np.abs(V)
    Va = np.angle(V)
    P, Q = polar_power_equations(Ybus, V)

    pv = np.asarray(pv, dtype=int)
    pq = np.asarray(pq, dtype=int)
    pvpq = np.r_[pv, pq]
    n_angle = len(pvpq)
    n_pq = len(pq)
    J = np.zeros((n_angle + n_pq, n_angle + n_pq), dtype=float)

    # J11 = ∂P/∂theta，行和列都是 [PV, PQ]。
    for row, i in enumerate(pvpq):
        for col, k in enumerate(pvpq):
            if i == k:
                J[row, col] = -Q[i] - B[i, i] * Vm[i] ** 2
            else:
                theta_ik = Va[i] - Va[k]
                J[row, col] = Vm[i] * Vm[k] * (
                    G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
                )

    # J12 = ∂P/∂Vm，行是 [PV, PQ]，列是 PQ。
    for row, i in enumerate(pvpq):
        for col, k in enumerate(pq):
            if i == k:
                J[row, n_angle + col] = P[i] / Vm[i] + G[i, i] * Vm[i]
            else:
                theta_ik = Va[i] - Va[k]
                J[row, n_angle + col] = Vm[i] * (
                    G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik)
                )

    # J21 = ∂Q/∂theta，行是 PQ，列是 [PV, PQ]。
    for row, i in enumerate(pq):
        for col, k in enumerate(pvpq):
            if i == k:
                J[n_angle + row, col] = P[i] - G[i, i] * Vm[i] ** 2
            else:
                theta_ik = Va[i] - Va[k]
                J[n_angle + row, col] = -Vm[i] * Vm[k] * (
                    G[i, k] * np.cos(theta_ik) + B[i, k] * np.sin(theta_ik)
                )

    # J22 = ∂Q/∂Vm，行和列都是 PQ。
    for row, i in enumerate(pq):
        for col, k in enumerate(pq):
            if i == k:
                J[n_angle + row, n_angle + col] = Q[i] / Vm[i] - B[i, i] * Vm[i]
            else:
                theta_ik = Va[i] - Va[k]
                J[n_angle + row, n_angle + col] = Vm[i] * (
                    G[i, k] * np.sin(theta_ik) - B[i, k] * np.cos(theta_ik)
                )

    return J


def solve_newton_raphson(
    problem: "PowerFlowProblem",
    tol: float = 1e-10,
    max_iter: int = 20,
) -> NewtonResult:
    """求解 AC 潮流并保存每次迭代的完整轨迹。"""

    if tol <= 0:
        raise ValueError("tol 必须为正数")
    if max_iter < 0:
        raise ValueError("max_iter 不能为负数")

    V = np.asarray(problem.V0, dtype=complex).copy()
    pv = np.asarray(problem.pv, dtype=int)
    pq = np.asarray(problem.pq, dtype=int)
    npv = len(pv)
    npq = len(pq)

    mismatch = mismatch_vector(problem.Ybus, problem.Sbus, V, pv, pq)
    norm = float(np.linalg.norm(mismatch, ord=np.inf))
    initial_mismatch = mismatch.copy()
    initial_norm = norm
    steps: list[NewtonStep] = []

    if norm < tol:
        return NewtonResult(
            voltage=V,
            converged=True,
            iterations=0,
            initial_mismatch=initial_mismatch,
            initial_norm=initial_norm,
            final_mismatch=mismatch.copy(),
            final_norm=norm,
            steps=steps,
        )

    converged = False
    for iteration in range(1, max_iter + 1):
        P, Q = polar_power_equations(problem.Ybus, V)
        J = jacobian_polar(problem.Ybus, V, pv, pq)

        # 线性方程 J Δx = -F。小系统用稠密 LU；生产代码通常使用稀疏分解。
        correction = np.linalg.solve(J, -mismatch)

        # Δx = [Δtheta_PV, Δtheta_PQ, ΔVm_PQ]
        Vm = np.abs(V)
        Va = np.angle(V)
        Va[pv] += correction[:npv]
        Va[pq] += correction[npv : npv + npq]
        Vm[pq] += correction[npv + npq :]
        V = Vm * np.exp(1j * Va)

        mismatch_after = mismatch_vector(problem.Ybus, problem.Sbus, V, pv, pq)
        norm_after = float(np.linalg.norm(mismatch_after, ord=np.inf))

        steps.append(
            NewtonStep(
                iteration=iteration,
                mismatch_before=mismatch.copy(),
                norm_before=norm,
                P=P.copy(),
                Q=Q.copy(),
                jacobian=J.copy(),
                correction=correction.copy(),
                max_correction=float(np.linalg.norm(correction, ord=np.inf)),
                voltage_after=V.copy(),
                mismatch_after=mismatch_after.copy(),
                norm_after=norm_after,
            )
        )

        mismatch = mismatch_after
        norm = norm_after
        if norm < tol:
            converged = True
            break

    return NewtonResult(
        voltage=V,
        converged=converged,
        iterations=len(steps),
        initial_mismatch=initial_mismatch,
        initial_norm=initial_norm,
        final_mismatch=mismatch.copy(),
        final_norm=norm,
        steps=steps,
    )
