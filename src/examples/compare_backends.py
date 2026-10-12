"""比较自研求解器、PYPOWER 和 Pandapower 的结果。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow import (  # noqa: E402
    run_pandapower_case9,
    run_pypower_case9,
    solve_case9,
)
from powerflow.report import (  # noqa: E402
    print_generator_results,
    print_iteration_summary,
    print_power_balance,
    print_voltage_comparison,
)
from pypower.idx_bus import VA, VM  # noqa: E402


def main() -> None:
    _, result, custom = solve_case9(tol=1e-10)
    pypower, success = run_pypower_case9(tol=1e-10)
    pandapower = run_pandapower_case9(tol_mva=1e-10)

    print("自研牛顿法迭代：")
    print_iteration_summary(result)
    print(f"\nPYPOWER success = {success}")
    print(f"Pandapower converged = {pandapower.converged}, "
          f"iterations = {pandapower._ppc.get('iterations')}")

    print("\n母线电压对照：")
    print_voltage_comparison(custom, pypower, pandapower)

    custom_vm = custom["bus"][:, VM]
    custom_va = custom["bus"][:, VA]
    pp_vm = pypower["bus"][:, VM]
    pp_va = pypower["bus"][:, VA]
    ap_vm = pandapower.res_bus["vm_pu"].to_numpy()
    ap_va = pandapower.res_bus["va_degree"].to_numpy()
    print("\n最大差异：")
    print(f"custom vs PYPOWER:  Vm={np.max(np.abs(custom_vm - pp_vm)):.3e}, "
          f"Va={np.max(np.abs(custom_va - pp_va)):.3e}")
    print(f"custom vs Pandapower: Vm={np.max(np.abs(custom_vm - ap_vm)):.3e}, "
          f"Va={np.max(np.abs(custom_va - ap_va)):.3e}")

    print("\n自研求解器回算的发电机结果：")
    print_generator_results(custom)
    print("\n功率平衡：")
    print_power_balance(custom)


if __name__ == "__main__":
    main()
