"""环境自检：验证 PYPOWER 和自研牛顿法结果一致。"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow import (  # noqa: E402
    run_pypower_case9,
    solve_case9,
)
from pypower.idx_bus import VA, VM  # noqa: E402
from pypower.idx_gen import PG  # noqa: E402


def main() -> None:
    print("PYPOWER   :", importlib.metadata.version("pypower"))

    _, custom_result, custom = solve_case9()
    pypower, success = run_pypower_case9()

    assert custom_result.converged
    assert success

    pp_vm = pypower["bus"][:, VM]
    pp_va = pypower["bus"][:, VA]
    custom_vm = custom["bus"][:, VM]
    custom_va = custom["bus"][:, VA]

    assert np.max(np.abs(custom_vm - pp_vm)) < 1e-9
    assert np.max(np.abs(custom_va - pp_va)) < 1e-7
    assert abs(float(np.sum(custom["gen"][:, PG])) - float(np.sum(pypower["gen"][:, PG]))) < 1e-8

    print(f"自研牛拉法: 收敛={custom_result.converged}, 迭代={custom_result.iterations}, "
          f"||F||inf={custom_result.final_norm:.3e}")
    print("PYPOWER   : 收敛=", bool(success))
    print("[PASS] Python 潮流计算环境正常")


if __name__ == "__main__":
    main()
