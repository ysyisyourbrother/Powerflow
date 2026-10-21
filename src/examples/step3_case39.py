"""IEEE 39 节点 New England 系统潮流计算完整示例。

一个算例，两条求解路径，互相验证：

1. PYPOWER ``runpf``          —— 成熟参考实现（推荐首选）
2. 项目自研极坐标牛顿法       —— 逐次迭代完全可观察，用于学习

运行::

    MPLCONFIGDIR=/tmp/mpl .venv/bin/python src/examples/step3_case39.py
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow import (  # noqa: E402
    pad_branch_columns,
    prepare_case,
    solve_newton_raphson,
)
from powerflow.plotting import plot_power_flow  # noqa: E402
from pypower.api import (  # noqa: E402
    case39,
    int2ext,
    pfsoln,
    ppoption,
    runpf,
)
from pypower.idx_brch import (  # noqa: E402
    BR_STATUS,
    F_BUS,
    PF,
    PT,
    QF,
    QT,
    RATE_A,
    T_BUS,
)
from pypower.idx_bus import (  # noqa: E402
    BASE_KV,
    BUS_I,
    BUS_TYPE,
    PD,
    PQ,
    PV,
    QD,
    REF,
    VA,
    VM,
)
from pypower.idx_gen import (  # noqa: E402
    GEN_BUS,
    PG,
    PMAX,
    PMIN,
    QG,
    QMAX,
    QMIN,
    VG,
)

TOL = 1e-10
MAX_IT = 20


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
# ``pad_branch_columns`` 与 ``prepare_case`` 已提升进 powerflow 包，
# 这里直接复用，避免 examples 与包内实现漂移。


def separator(title: str = "") -> None:
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


# --------------------------------------------------------------------------
# 1. 算例数据概览
# --------------------------------------------------------------------------
def show_case_data(ppc: dict) -> None:
    separator("第 1 步：读入 IEEE 39 节点算例（PYPOWER case39）")

    bus, gen, branch = ppc["bus"], ppc["gen"], ppc["branch"]
    types = bus[:, BUS_TYPE].astype(int)
    load_mw = float(np.sum(bus[:, PD]))
    load_mvar = float(np.sum(bus[:, QD]))

    print(f"baseMVA          : {ppc['baseMVA']:.1f} MVA")
    print(f"母线 bus         : {bus.shape[0]} 个")
    print(f"发电机 gen       : {gen.shape[0]} 台")
    print(f"支路 branch      : {branch.shape[0]} 条"
          f"（线路 35 + 变压器 11）")
    print(f"总负荷           : {load_mw:.2f} MW + j{load_mvar:.2f} Mvar")
    print(f"母线类型         : PQ {int(np.sum(types == PQ))} 个 / "
          f"PV {int(np.sum(types == PV))} 个 / "
          f"平衡 {int(np.sum(types == REF))} 个")

    print("\n发电机一览：")
    print(f"{'母线':>4} {'P 设定(MW)':>11} {'V 设定(pu)':>11} "
          f"{'Qmin(Mvar)':>11} {'Qmax(Mvar)':>11} {'Pmin':>7} {'Pmax':>7}")
    for row in gen:
        print(f"{int(row[GEN_BUS]):>4} {row[PG]:>11.2f} {row[VG]:>11.4f} "
              f"{row[QMIN]:>11.1f} {row[QMAX]:>11.1f} "
              f"{row[PMIN]:>7.0f} {row[PMAX]:>7.0f}")

    print("\nPQ 母线（负荷节点）一览：")
    print(f"{'母线':>4} {'Pd(MW)':>9} {'Qd(Mvar)':>9} {'基准kV':>8}")
    for row in bus:
        if int(row[BUS_TYPE]) == PQ and (row[PD] != 0 or row[QD] != 0):
            print(f"{int(row[BUS_I]):>4} {row[PD]:>9.1f} {row[QD]:>9.1f} "
                  f"{row[BASE_KV]:>8.0f}")


# --------------------------------------------------------------------------
# 2. 求解路径 A：PYPOWER
# --------------------------------------------------------------------------
def solve_with_pypower(ppc: dict) -> tuple[dict, bool]:
    separator("第 2 步：求解路径 A —— PYPOWER runpf（牛顿-拉夫逊法）")

    options = ppoption(
        VERBOSE=1,
        OUT_ALL=0,
        PF_ALG=1,        # 1 = 极坐标牛顿法
        PF_TOL=TOL,
        PF_MAX_IT=MAX_IT,
    )
    result, success = runpf(pad_branch_columns(ppc), options)

    print(f"\n收敛标志 success = {success}")
    if success:
        vm = result["bus"][:, VM]
        va = result["bus"][:, VA]
        print(f"母线电压幅值范围 : {vm.min():.4f} ~ {vm.max():.4f} pu")
        print(f"母线电压相角范围 : {va.min():.4f}° ~ {va.max():.4f}°")
    return result, bool(success)


# --------------------------------------------------------------------------
# 3. 求解路径 B：自研极坐标牛顿法（带迭代轨迹）
# --------------------------------------------------------------------------
def solve_with_custom_newton(ppc: dict):
    separator("第 3 步：求解路径 B —— 自研极坐标牛顿法（可观察每次迭代）")

    problem = prepare_case(ppc)
    result = solve_newton_raphson(problem, tol=TOL, max_iter=MAX_IT)

    print(f"未知量维度 : {problem.nb * 2 - len(problem.pv) - 2} "
          f"（每母线 2 个 Vm/θ，去掉 PV 的 Vm 和平衡母线的 Vm、θ）")
    print(f"Ybus 规模  : {problem.Ybus.shape}, "
          f"非零元 {problem.Ybus.nnz if hasattr(problem.Ybus, 'nnz') else '-'}")
    print(f"\n{'迭代':>4} {'迭代前失配范数':>18} {'最大修正量':>16} {'迭代后失配范数':>18}")
    for step in result.steps:
        print(f"{step.iteration:>4} {step.norm_before:>18.3e} "
              f"{step.max_correction:>16.3e} {step.norm_after:>18.3e}")
    print(f"\n收敛 = {result.converged}，迭代次数 = {result.iterations}")

    # 用最终电压回算发电机出力与支路潮流
    bus = problem.ppci["bus"].copy()
    gen = problem.ppci["gen"].copy()
    branch = problem.ppci["branch"].copy()
    bus, gen, branch = pfsoln(
        problem.base_mva, bus, gen, branch,
        problem.Ybus, problem.Yf, problem.Yt,
        result.voltage, problem.ref, problem.pv, problem.pq,
    )
    solved = dict(problem.ppci)
    solved.update({"bus": bus, "gen": gen, "branch": branch,
                   "success": int(result.converged),
                   "iterations": result.iterations})
    external = int2ext(solved)
    external["iterations"] = result.iterations
    return problem, result, external


# --------------------------------------------------------------------------
# 3b. 平启动对照实验
# --------------------------------------------------------------------------
def flat_start_experiment(ppc: dict):
    separator("第 3b 步：平启动（flat start）对照实验")

    print("case39 自带的 bus.VM / bus.VA 就是厂家给出的已收敛解，")
    print("所以直接求解只要 1 次迭代。把初值改成 Vm=1.0、Va=0 的平启动，")
    print("才能看到牛顿法真正的迭代收敛过程。\n")

    flat = copy.deepcopy(ppc)
    flat["bus"] = np.asarray(flat["bus"], dtype=float)
    flat["bus"][:, VM] = 1.0
    flat["bus"][:, VA] = 0.0

    problem = prepare_case(flat)
    result = solve_newton_raphson(problem, tol=TOL, max_iter=MAX_IT)

    print(f"{'迭代':>4} {'迭代前失配范数':>18} {'最大修正量':>16} "
          f"{'迭代后失配范数':>18} {'下降倍数':>12}")
    previous = None
    for step in result.steps:
        ratio = f"{previous / step.norm_after:>12.1e}" if previous else f"{'-':>12}"
        print(f"{step.iteration:>4} {step.norm_before:>18.6e} "
              f"{step.max_correction:>16.6e} {step.norm_after:>18.6e} {ratio}")
        previous = step.norm_after

    print(f"\n平启动收敛 = {result.converged}，迭代 {result.iterations} 次")
    vm = np.abs(result.voltage)
    print(f"平启动最终电压范围 = {vm.min():.4f} ~ {vm.max():.4f} pu")
    ref = np.asarray(ppc["bus"], dtype=float)[:, VM]
    print(f"与随算例初值所求结果的最大电压差 = {np.max(np.abs(vm - ref)):.3e} pu")
    return result


def plot_convergence(result, outfile: Path) -> Path:
    """画出平启动牛顿迭代的收敛曲线（半对数坐标）。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 让图里的中文能正常显示（macOS 自带中文字体）
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Hiragino Sans GB",
        "Arial Unicode MS", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    iterations = [step.iteration for step in result.steps]
    norms = [step.norm_after for step in result.steps]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.semilogy(iterations, norms, "o-", color="#1f77b4", lw=2, ms=7,
                label="迭代后失配范数 $\\|\\Delta S\\|_\\infty$")
    for x, y in zip(iterations, norms):
        ax.annotate(f"{y:.2e}", (x, y), textcoords="offset points",
                    xytext=(6, 8), fontsize=8)
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("最大功率失配（p.u.，对数坐标）")
    ax.set_title("IEEE 39 节点：平启动牛顿法的二次收敛")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_xticks(iterations)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return Path(outfile)


def reactive_balance(result: dict) -> None:
    """拆解网络无功平衡：线路充电发出的无功 vs 串联电抗消耗的无功。"""

    separator("第 5 步：无功平衡拆解（为什么网损无功是负的）")

    bus, branch = result["bus"], result["branch"]
    vm = bus[:, VM]
    f = branch[:, F_BUS].astype(int) - 1
    t = branch[:, T_BUS].astype(int) - 1
    b = branch[:, 4]

    # 线路充电功率 Q_c = (b/2) * (Vm_f^2 + Vm_t^2)，变压器 b=0
    q_charging = float(np.sum(0.5 * b * (vm[f] ** 2 + vm[t] ** 2))) * 100.0
    q_branch_loss = float(np.sum(branch[:, QF] + branch[:, QT]))

    print(f"线路充电注入无功（电容效应）  = {q_charging:>10.2f} Mvar")
    print(f"支路总无功损耗 Σ(Qf+Qt)       = {q_branch_loss:>10.2f} Mvar")
    print(f"串联电抗消耗无功（推算）      = {q_charging - q_branch_loss:>10.2f} Mvar")
    print()
    print("说明：39 节点系统的线路充电电纳很大（Σb = 10.36 pu），")
    print("电容发出的无功多于串联电抗消耗的无功，")
    print("所以支路总无功损耗为负——网络整体是无功电源而不是无功负荷。")


# --------------------------------------------------------------------------
# 4. 结果展示
# --------------------------------------------------------------------------
def show_results(result: dict) -> None:
    separator("第 4 步：结果解读")

    bus, gen, branch = result["bus"], result["gen"], result["branch"]
    base = float(result["baseMVA"])

    print("【母线结果】VM 电压幅值，VA 电压相角，PD/QD 负荷")
    print(f"{'母线':>4} {'类型':>4} {'Vm(pu)':>9} {'Va(°)':>9} "
          f"{'Pd(MW)':>9} {'Qd(Mvar)':>9}")
    for row in bus:
        print(f"{int(row[BUS_I]):>4} {int(row[BUS_TYPE]):>4} "
              f"{row[VM]:>9.4f} {row[VA]:>9.4f} {row[PD]:>9.1f} {row[QD]:>9.1f}")

    print("\n【发电机结果】实际有功/无功出力，是否越限")
    print(f"{'母线':>4} {'Pg(MW)':>9} {'Qg(Mvar)':>10} "
          f"{'Qmin':>8} {'Qmax':>8}  限值")
    for row in gen:
        flag = ""
        if row[QG] > row[QMAX] + 1e-6:
            flag = "← Q 越上限"
        elif row[QG] < row[QMIN] - 1e-6:
            flag = "← Q 越下限"
        print(f"{int(row[GEN_BUS]):>4} {row[PG]:>9.2f} {row[QG]:>10.2f} "
              f"{row[QMIN]:>8.1f} {row[QMAX]:>8.1f}  {flag}")

    print("\n【支路潮流】只列负载率最高的 12 条")
    rows = []
    for row in branch:
        if row[BR_STATUS] <= 0:
            continue
        rating = float(row[RATE_A])
        flow = max(abs(float(row[PF])), abs(float(row[PT])))
        loading = 100.0 * flow / rating if rating > 0 else 0.0
        rows.append((loading, row, flow))
    rows.sort(key=lambda item: -item[0])
    print(f"{'from':>4} {'to':>4} {'P_from(MW)':>11} {'Q_from(Mvar)':>13} "
          f"{'P_to(MW)':>10} {'容量(MVA)':>9} {'负载率%':>8}")
    for loading, row, flow in rows[:12]:
        print(f"{int(row[F_BUS]):>4} {int(row[T_BUS]):>4} "
              f"{row[PF]:>11.2f} {row[QF]:>13.2f} {row[PT]:>10.2f} "
              f"{row[RATE_A]:>9.0f} {loading:>8.2f}")

    print("\n【功率平衡】")
    gen_mw = float(np.sum(gen[:, PG]))
    load_mw = float(np.sum(bus[:, PD]))
    loss_mw = gen_mw - load_mw
    loss_branch = float(np.sum(branch[:, PF] + branch[:, PT]))
    gen_mvar = float(np.sum(gen[:, QG]))
    load_mvar = float(np.sum(bus[:, QD]))
    loss_mvar = float(np.sum(branch[:, QF] + branch[:, QT]))
    print(f"总发电有功  ΣPg = {gen_mw:>10.4f} MW")
    print(f"总负荷有功  ΣPd = {load_mw:>10.4f} MW")
    print(f"网络有功损耗     = {loss_mw:>10.4f} MW"
          f"   （支路累加校验 {loss_branch:.4f} MW）")
    print(f"总发电无功  ΣQg = {gen_mvar:>10.4f} Mvar")
    print(f"总负荷无功  ΣQd = {load_mvar:>10.4f} Mvar")
    print(f"网络无功损耗     = {gen_mvar - load_mvar:>10.4f} Mvar"
          f"   （支路累加校验 {loss_mvar:.4f} Mvar）")

    # 用标幺值再验算一次
    print("\n【功率平衡（标幺值）】")
    print(f"ΣPg = {gen_mw / base:.6f} pu, ΣPd = {load_mw / base:.6f} pu, "
          f"损耗 = {loss_mw / base:.6f} pu")


def compare_backends(pypower_result: dict, custom_external: dict) -> None:
    separator("第 6 步：两条路径结果对照")

    pp_bus = pypower_result["bus"]
    cm_bus = custom_external["bus"]

    vm_pp = pp_bus[:, VM]
    vm_cm = cm_bus[:, VM]

    va_pp = pp_bus[:, VA]
    va_cm = cm_bus[:, VA]

    order = np.argsort(pp_bus[:, BUS_I].astype(int))
    print("PYPOWER  vs 自研牛顿法：")
    print(f"  电压幅值最大差异 = {np.max(np.abs(vm_pp[order] - vm_cm[order])):.3e} pu")
    print(f"  电压相角最大差异 = {np.max(np.abs(va_pp[order] - va_cm[order])):.3e}°")


# --------------------------------------------------------------------------
def main() -> None:
    ppc = case39()
    show_case_data(ppc)

    pypower_result, success = solve_with_pypower(ppc)
    problem, custom_result, custom_external = solve_with_custom_newton(ppc)
    flat_result = flat_start_experiment(ppc)

    show_results(pypower_result)
    reactive_balance(pypower_result)
    compare_backends(pypower_result, custom_external)

    outfile = Path(__file__).with_name("case39_result.png")
    saved = plot_power_flow(
        pypower_result, outfile, title="IEEE 39-bus New England power flow"
    )
    conv_file = plot_convergence(
        flat_result, Path(__file__).with_name("case39_convergence.png")
    )
    separator("完成")
    print(f"PYPOWER 收敛: {success}")
    print(f"自研求解器  : 收敛 {custom_result.converged}, "
          f"{custom_result.iterations} 次迭代")
    print(f"网络图已保存: {saved}")
    print(f"收敛曲线    : {conv_file}")


if __name__ == "__main__":
    main()
