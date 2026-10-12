import numpy as np

from powerflow import jacobian_polar, mismatch_vector, prepare_case9


def _voltage_from_x(problem, x0, x):
    Va = np.angle(problem.V0).copy()
    Vm = np.abs(problem.V0).copy()
    npv = len(problem.pv)
    npq = len(problem.pq)
    Va[problem.pv] = x[:npv]
    Va[problem.pq] = x[npv : npv + npq]
    Vm[problem.pq] = x[npv + npq :]
    assert x0.shape == x.shape
    return Vm * np.exp(1j * Va)


def test_polar_jacobian_matches_central_difference():
    problem = prepare_case9()
    Va0 = np.angle(problem.V0)
    Vm0 = np.abs(problem.V0)
    x0 = np.r_[Va0[problem.pv], Va0[problem.pq], Vm0[problem.pq]]
    J = jacobian_polar(problem.Ybus, problem.V0, problem.pv, problem.pq)
    step = 1e-6
    max_error = 0.0

    for column in range(len(x0)):
        xp = x0.copy()
        xm = x0.copy()
        xp[column] += step
        xm[column] -= step
        Vp = _voltage_from_x(problem, x0, xp)
        Vm = _voltage_from_x(problem, x0, xm)
        Fp = mismatch_vector(problem.Ybus, problem.Sbus, Vp, problem.pv, problem.pq)
        Fm = mismatch_vector(problem.Ybus, problem.Sbus, Vm, problem.pv, problem.pq)
        numerical = (Fp - Fm) / (2.0 * step)
        max_error = max(max_error, float(np.max(np.abs(J[:, column] - numerical))))

    assert max_error < 2e-8
