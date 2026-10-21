import numpy as np
from pypower.api import dSbus_dV
from scipy.sparse import hstack, vstack

from powerflow import (
    build_external_result,
    jacobian_polar,
    prepare_case9,
    run_pypower_case9,
    solve_newton_raphson,
)
from pypower.idx_bus import VA, VM


def test_case9_converges_in_four_iterations():
    problem = prepare_case9()
    result = solve_newton_raphson(problem, tol=1e-10)
    assert result.converged
    assert result.iterations == 4
    assert result.final_norm < 1e-10


def test_jacobian_matches_pypower_complex_derivative():
    problem = prepare_case9()
    J = jacobian_polar(problem.Ybus, problem.V0, problem.pv, problem.pq)
    pvpq = np.r_[problem.pv, problem.pq]
    dS_dVm, dS_dVa = dSbus_dV(problem.Ybus, problem.V0)
    J11 = dS_dVa[pvpq][:, pvpq].real
    J12 = dS_dVm[pvpq][:, problem.pq].real
    J21 = dS_dVa[problem.pq][:, pvpq].imag
    J22 = dS_dVm[problem.pq][:, problem.pq].imag
    reference = vstack([hstack([J11, J12]), hstack([J21, J22])]).toarray()
    assert np.max(np.abs(J - reference)) < 1e-12


def test_custom_matches_pypower():
    problem = prepare_case9()
    result = solve_newton_raphson(problem)
    custom = build_external_result(problem, result)
    pypower, success = run_pypower_case9()

    assert success
    assert np.max(np.abs(custom["bus"][:, VM] - pypower["bus"][:, VM])) < 1e-9
    assert np.max(np.abs(custom["bus"][:, VA] - pypower["bus"][:, VA])) < 1e-7
