"""导出 case39 潮流计算结果表（markdown 片段）。"""
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "src/examples")

import numpy as np
from pypower.api import case39, ppoption, runpf

from step3_case39 import pad_branch_columns

res, ok = runpf(
    pad_branch_columns(case39()),
    ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1, PF_TOL=1e-10),
)
bus, gen, br = res["bus"], res["gen"], res["branch"]

print("### BUS SOLUTION")
print("| bus | 类型 | Vm(pu) | Va(deg) | Pd(MW) | Qd(Mvar) | 计算 Pg(MW) | 计算 Qg(Mvar) |")
print("|---:|:--|---:|---:|---:|---:|---:|---:|")
tname = {1: "PQ", 2: "PV", 3: "平衡"}
for r in bus:
    bi = int(r[0])
    # 该母线发电机的注入
    mask = gen[:, 0].astype(int) == bi
    pg = float(np.sum(gen[mask, 1])) if mask.any() else 0.0
    qg = float(np.sum(gen[mask, 2])) if mask.any() else 0.0
    print(
        "| %d | %s | %.4f | %.4f | %g | %g | %s | %s |"
        % (
            bi, tname[int(r[1])], r[7], r[8], r[2], r[3],
            f"{pg:.2f}" if mask.any() else "-",
            f"{qg:.2f}" if mask.any() else "-",
        )
    )

print()
print("### GEN SOLUTION")
print("| 母线 | Pg(MW) | Qg(Mvar) | Qmin | Qmax | Q 是否越限 | Pmax | P 是否越限 |")
print("|---:|---:|---:|---:|---:|:--|---:|:--|")
for r in gen:
    qflag = "越上限" if r[2] > r[3] + 1e-6 else ("越下限" if r[2] < r[4] - 1e-6 else "正常")
    pflag = "越上限" if r[1] > r[8] + 1e-6 else ("越下限" if r[1] < r[9] - 1e-6 else "正常")
    print(
        "| %d | %.2f | %.2f | %g | %g | %s | %g | %s |"
        % (r[0], r[1], r[2], r[4], r[3], qflag, r[8], pflag)
    )

print()
print("### BRANCH SOLUTION")
print("| # | from | to | P_from(MW) | Q_from(Mvar) | P_to(MW) | Q_to(Mvar) | 损耗P(MW) | 容量 | 负载率% |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for i, r in enumerate(br, 1):
    flow = max(abs(r[13]), abs(r[15]))
    loading = 100.0 * flow / r[5] if r[5] > 0 else 0.0
    loss = r[13] + r[15]
    print(
        "| %d | %d | %d | %.2f | %.2f | %.2f | %.2f | %.3f | %g | %.2f |"
        % (i, r[0], r[1], r[13], r[14], r[15], r[16], loss, r[5], loading)
    )

print()
print("### 汇总")
print("Pg total", float(np.sum(gen[:, 1])))
print("Pd total", float(np.sum(bus[:, 2])))
print("loss P", float(np.sum(gen[:, 1])) - float(np.sum(bus[:, 2])))
print("Qg total", float(np.sum(gen[:, 2])))
print("Qd total", float(np.sum(bus[:, 3])))
print("loss Q", float(np.sum(gen[:, 2])) - float(np.sum(bus[:, 3])))
print("vm min/max", float(bus[:, 7].min()), float(bus[:, 7].max()))
print("vm min bus", int(bus[np.argmin(bus[:, 7]), 0]), "vm max bus", int(bus[np.argmax(bus[:, 7]), 0]))
print("va min/max", float(bus[:, 8].min()), float(bus[:, 8].max()))
print("va min bus", int(bus[np.argmin(bus[:, 8]), 0]), "va max bus", int(bus[np.argmax(bus[:, 8]), 0]))
loadings = np.array([
    100.0 * max(abs(r[13]), abs(r[15])) / r[5] for r in br
])
print("max loading %.2f%% at branch %d (%d-%d)" % (loadings.max(), loadings.argmax() + 1, br[loadings.argmax(), 0], br[loadings.argmax(), 1]))
print("mean loading %.2f%%" % loadings.mean())
print("branches over 60%%:", [i + 1 for i, l in enumerate(loadings) if l > 60])
print("total loss P from branches", float(np.sum(br[:, 13] + br[:, 15])))
print("total loss Q from branches", float(np.sum(br[:, 14] + br[:, 16])))
