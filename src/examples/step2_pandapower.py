"""Pandapower 示例：把同一个 PYPOWER case9 转成 Pandapower 网络。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow import run_pandapower_case9  # noqa: E402


def main() -> None:
    net = run_pandapower_case9()
    assert net.converged

    print("Pandapower 算例")
    print(f"母线: {len(net.bus)}, 负荷: {len(net.load)}, 发电机: {len(net.gen)}, "
          f"外部电网: {len(net.ext_grid)}, 线路: {len(net.line)}")
    print(f"是否收敛: {net.converged}")
    print(f"迭代次数: {net._ppc.get('iterations')}")
    print("\n母线结果：")
    print(net.res_bus[["vm_pu", "va_degree", "p_mw", "q_mvar"]].to_string())
    print("\n发电机结果：")
    print(net.res_gen[["p_mw", "q_mvar", "vm_pu"]].to_string())
    print("\n平衡发电机结果：")
    print(net.res_ext_grid[["p_mw", "q_mvar"]].to_string())
    print("\n线路结果：")
    print(
        net.res_line[
            ["p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar", "loading_percent"]
        ].to_string()
    )
    total_generation = float(net.res_ext_grid["p_mw"].sum() + net.res_gen["p_mw"].sum())
    print(f"\n总发电: {total_generation:.6f} MW")
    print(f"总负荷: {net.load['p_mw'].sum():.6f} MW")
    print(f"网络损耗: {total_generation - net.load['p_mw'].sum():.6f} MW")


if __name__ == "__main__":
    main()
