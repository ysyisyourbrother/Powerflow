"""导出 case39 教学文档所需的全部数据表（markdown 片段）。

从仓库任意位置运行均可，脚本会自行定位仓库根目录下的 ``src/``。
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from pypower.api import case39

ppc = case39()
bus, gen, br = ppc["bus"], ppc["gen"], ppc["branch"]

print("### BUS TABLE")
print("| bus | type | Pd(MW) | Qd(Mvar) | Gs(MW) | Bs(Mvar) | area | Vm(pu) | Va(deg) | baseKV | Vmax | Vmin |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in bus:
    print(
        "| %d | %d | %g | %g | %g | %g | %d | %g | %g | %g | %g | %g |"
        % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[11], r[12])
    )

print()
print("### GEN TABLE")
print("| bus | Pg(MW) | Qg(Mvar) | Qmax | Qmin | Vg(pu) | mBase | status | Pmax | Pmin |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in gen:
    print(
        "| %d | %g | %g | %g | %g | %g | %g | %d | %g | %g |"
        % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9])
    )

print()
print("### BRANCH TABLE")
print("| # | from | to | r(pu) | x(pu) | b(pu) | rateA(MVA) | tap | shift(deg) | 类型 |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|")
for i, r in enumerate(br, 1):
    kind = "变压器" if (r[8] not in (0.0, 1.0) or r[9] != 0.0) else "线路"
    print(
        "| %d | %d | %d | %g | %g | %g | %g | %g | %g | %s |"
        % (i, r[0], r[1], r[2], r[3], r[4], r[5], r[8], r[9], kind)
    )

print()
print("### 统计")
types = bus[:, 1].astype(int)
print("PQ", int(np.sum(types == 1)), "PV", int(np.sum(types == 2)), "REF", int(np.sum(types == 3)))
print("Pd total", float(np.sum(bus[:, 2])), "Qd total", float(np.sum(bus[:, 3])))
print("Pg init total", float(np.sum(gen[:, 1])), "Qg init total", float(np.sum(gen[:, 2])))
print("Gs total", float(np.sum(bus[:, 4])), "Bs total", float(np.sum(bus[:, 5])))
is_trafo = (br[:, 8] != 0) & ((br[:, 8] != 1.0) | (br[:, 9] != 0))
print("transformers", int(np.sum(is_trafo)), "lines", int(np.sum(~is_trafo)))
print("total b (charging)", float(np.sum(br[:, 4])))
print("total series R loss param", float(np.sum(br[:, 2])))
print("buses with load:", [int(r[0]) for r in bus if r[2] != 0 or r[3] != 0])
print("bus 31 row:", bus[30].tolist())
print("bus 39 row:", bus[38].tolist())
print("gen buses:", [int(r[0]) for r in gen])
print("REF bus:", [int(r[0]) for r in bus if r[1] == 3])
print("PV buses:", [int(r[0]) for r in bus if r[1] == 2])
print("PQ buses:", [int(r[0]) for r in bus if r[1] == 1])
print("vmax overall", float(np.max(bus[:, 11])), "vmin overall", float(np.min(bus[:, 12])))
print("rateA min", float(np.min(br[:, 5])), "max", float(np.max(br[:, 5])))
