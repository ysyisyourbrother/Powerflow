"""导出本机 PYPOWER 自带的 IEEE 39 节点原始数据和参考解。

运行：

    .venv/bin/python dataset/ieee39/scripts/export_ieee39.py

该脚本不修改源数据，只把 PYPOWER 中的数据表和求解结果写入 dataset/ieee39/。
"""

from __future__ import annotations

import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pypower.api import case39 as pypower_case39
from pypower.api import ppoption, runpf
from pypower.idx_brch import (
    ANGMAX,
    ANGMIN,
    BR_B,
    BR_R,
    BR_STATUS,
    BR_X,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    RATE_B,
    RATE_C,
    SHIFT,
    T_BUS,
    TAP,
)
from pypower.idx_bus import (
    BUS_AREA,
    BASE_KV,
    BS,
    BUS_I,
    BUS_TYPE,
    GS,
    PD,
    QD,
    VA,
    VM,
    VMAX,
    VMIN,
    ZONE,
)
from pypower.idx_gen import (
    APF,
    GEN_BUS,
    GEN_STATUS,
    MBASE,
    PC1,
    PC2,
    PG,
    PMAX,
    PMIN,
    QC1MAX,
    QC1MIN,
    QC2MAX,
    QC2MIN,
    QG,
    QMAX,
    QMIN,
    RAMP_10,
    RAMP_30,
    RAMP_AGC,
    RAMP_Q,
    VG,
)


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
REFERENCE = ROOT / "reference"
PYPOWER_RAW = RAW / "pypower"

PYPOWER_BUS_COLUMNS = [
    ("bus_i", BUS_I),
    ("bus_type", BUS_TYPE),
    ("pd_mw", PD),
    ("qd_mvar", QD),
    ("gs_mw", GS),
    ("bs_mvar", BS),
    ("area", BUS_AREA),
    ("vm_pu", VM),
    ("va_degree", VA),
    ("base_kv", BASE_KV),
    ("zone", ZONE),
    ("vmax_pu", VMAX),
    ("vmin_pu", VMIN),
]

PYPOWER_GEN_COLUMNS = [
    ("gen_bus", GEN_BUS),
    ("pg_mw", PG),
    ("qg_mvar", QG),
    ("qmax_mvar", QMAX),
    ("qmin_mvar", QMIN),
    ("vg_pu", VG),
    ("mbase_mva", MBASE),
    ("status", GEN_STATUS),
    ("pmax_mw", PMAX),
    ("pmin_mw", PMIN),
    ("pc1_mw", PC1),
    ("pc2_mw", PC2),
    ("qc1min_mvar", QC1MIN),
    ("qc1max_mvar", QC1MAX),
    ("qc2min_mvar", QC2MIN),
    ("qc2max_mvar", QC2MAX),
    ("ramp_agc_mw", RAMP_AGC),
    ("ramp_10_mw", RAMP_10),
    ("ramp_30_mw", RAMP_30),
    ("ramp_q_mvar", RAMP_Q),
    ("apf", APF),
]

PYPOWER_BRANCH_COLUMNS = [
    ("f_bus", F_BUS),
    ("t_bus", T_BUS),
    ("r_pu", BR_R),
    ("x_pu", BR_X),
    ("b_pu", BR_B),
    ("rate_a_mva", RATE_A),
    ("rate_b_mva", RATE_B),
    ("rate_c_mva", RATE_C),
    ("tap_pu", TAP),
    ("shift_degree", SHIFT),
    ("status", BR_STATUS),
    ("angmin_degree", ANGMIN),
    ("angmax_degree", ANGMAX),
]


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"无法序列化 {type(value)!r}")


def _records(array: np.ndarray, columns: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [
        {name: row[index].item() for name, index in columns}
        for row in array
    ]


def _write_csv(path: Path, array: np.ndarray, columns: list[tuple[str, int]]) -> None:
    frame = pd.DataFrame(
        {
            name: array[:, index]
            for name, index in columns
        }
    )
    frame.to_csv(path, index=False, float_format="%.15g")


def export_pypower() -> dict[str, Any]:
    PYPOWER_RAW.mkdir(parents=True, exist_ok=True)
    ppc = pypower_case39()

    bus_columns = PYPOWER_BUS_COLUMNS
    gen_columns = PYPOWER_GEN_COLUMNS
    branch_columns = PYPOWER_BRANCH_COLUMNS

    _write_csv(PYPOWER_RAW / "bus.csv", ppc["bus"], bus_columns)
    _write_csv(PYPOWER_RAW / "gen.csv", ppc["gen"], gen_columns)
    _write_csv(PYPOWER_RAW / "branch.csv", ppc["branch"], branch_columns)

    gencost = np.asarray(ppc["gencost"])
    pd.DataFrame(
        gencost,
        columns=["model", "startup_cost", "shutdown_cost", "ncost", "c2", "c1", "c0"],
    ).to_csv(
        PYPOWER_RAW / "gencost.csv", index=False, float_format="%.15g"
    )
    np.savez_compressed(
        PYPOWER_RAW / "pypower_case39_arrays.npz",
        baseMVA=np.asarray(ppc["baseMVA"]),
        bus=ppc["bus"],
        gen=ppc["gen"],
        branch=ppc["branch"],
        gencost=gencost,
    )

    payload = {
        "source": "pypower.case39",
        "source_module": inspect.getsourcefile(pypower_case39),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "baseMVA": ppc["baseMVA"],
        "version": ppc["version"],
        "bus": _records(ppc["bus"], bus_columns),
        "gen": _records(ppc["gen"], gen_columns),
        "branch": _records(ppc["branch"], branch_columns),
        "gencost": gencost.tolist(),
    }
    (PYPOWER_RAW / "case39.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return {"bus": len(ppc["bus"]), "gen": len(ppc["gen"]), "branch": len(ppc["branch"])}


def export_reference_solution() -> dict[str, Any]:
    REFERENCE.mkdir(parents=True, exist_ok=True)
    ppc = pypower_case39()
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-10, PF_MAX_IT=30)

    result, success = runpf(ppc, options)
    bus_solution = pd.DataFrame(
        result["bus"][:, [BUS_I, BUS_TYPE, VM, VA, PD, QD]],
        columns=["bus_i", "bus_type", "vm_pu", "va_degree", "pd_mw", "qd_mvar"],
    )
    bus_solution.to_csv(REFERENCE / "bus_solution_pypower.csv", index=False, float_format="%.12g")

    gen_solution = pd.DataFrame(
        result["gen"][:, [GEN_BUS, PG, QG, VG, GEN_STATUS]],
        columns=["gen_bus", "pg_mw", "qg_mvar", "vg_pu", "status"],
    )
    gen_solution.to_csv(REFERENCE / "gen_solution_pypower.csv", index=False, float_format="%.12g")

    branch_solution = pd.DataFrame(
        result["branch"][:, [F_BUS, T_BUS, PF, QF, PT, QT, RATE_A]],
        columns=["f_bus", "t_bus", "pf_mw", "qf_mvar", "pt_mw", "qt_mvar", "rate_a_mva"],
    )
    branch_solution.to_csv(
        REFERENCE / "branch_solution_pypower.csv", index=False, float_format="%.12g"
    )

    total_load = float(result["bus"][:, PD].sum())
    total_gen = float(result["gen"][:, PG].sum())
    summary = {
        "source": "PYPOWER case39",
        "network": "IEEE 39-bus New England system",
        "baseMVA": float(result["baseMVA"]),
        "buses": int(result["bus"].shape[0]),
        "generators": int(result["gen"].shape[0]),
        "branches": int(result["branch"].shape[0]),
        "load_mw": total_load,
        "load_mvar": float(result["bus"][:, QD].sum()),
        "generation_mw": total_gen,
        "loss_mw": total_gen - total_load,
        "success": bool(success),
        "vm_min_pu": float(result["bus"][:, VM].min()),
        "vm_max_pu": float(result["bus"][:, VM].max()),
        "va_min_degree": float(result["bus"][:, VA].min()),
        "va_max_degree": float(result["bus"][:, VA].max()),
    }
    (REFERENCE / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return summary


def write_checksums() -> None:
    lines = []
    for path in sorted(RAW.rglob("*")):
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            lines.append(f"{digest}  {path.relative_to(ROOT)}")
    (ROOT / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    pypower_counts = export_pypower()
    summary = export_reference_solution()
    write_checksums()
    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "pypower": pypower_counts,
        "reference_solution": summary,
        "raw_root": str(RAW.relative_to(ROOT)),
        "checksums": "checksums.sha256",
        "upstream_references": {
            "pypower_case39": "https://github.com/rwl/PYPOWER/blob/master/pypower/case39.py",
            "original_1970_report": "Bills et al., On-Line Stability Analysis Study, EEI RP90-1, 1970",
            "pai_1989": "M. A. Pai, Energy Function Analysis for Power System Stability, 1989",
            "athay_1979": "Athay, Podmore, Virmani, A Practical Method for the Direct Analysis of Transient Stability, 1979",
        },
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    print(f"导出完成：{ROOT}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
