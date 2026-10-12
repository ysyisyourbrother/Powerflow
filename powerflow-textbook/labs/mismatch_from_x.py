"""把扁平状态向量 x 映射回复电压并计算失配 F(x)。"""

from __future__ import annotations

import numpy as np


def voltage_from_x(
    x: np.ndarray,
    Va0: np.ndarray,
    Vm0: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
) -> np.ndarray:
    npv = len(pv)
    npq = len(pq)
    Va = Va0.copy()
    Vm = Vm0.copy()
    Va[pv] = x[:npv]
    Va[pq] = x[npv : npv + npq]
    Vm[pq] = x[npv + npq :]
    return Vm * np.exp(1j * Va)


def mismatch_from_x(
    x: np.ndarray,
    Ybus,
    Sbus: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    Va0: np.ndarray,
    Vm0: np.ndarray,
) -> np.ndarray:
    V = voltage_from_x(x, Va0, Vm0, pv, pq)
    delta_S = V * np.conj(Ybus.dot(V)) - Sbus
    return np.r_[delta_S[pv].real, delta_S[pq].real, delta_S[pq].imag]
