"""负荷扰动采样器。

当前比赛场景只启用一个扰动算子——负荷缩放。刻意不做发电调度置换：
一是保持问题良态，二是让平衡机的出力成为**真正未被观测的标签**，
而不是把答案泄漏进输入。

采样器是 ``(基准算例, 配置, 样本号)`` 的纯函数。随机数由
``SeedSequence([master_seed, sample_id])`` 派生，因此第 i 条样本可以
单独复现，无需重放前 i-1 条——分片或并行生成时这是刚需。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
from pypower.api import makeSbus
from pypower.idx_bus import PD, QD

from ..cases import external_bus_rows

if TYPE_CHECKING:  # pragma: no cover - 仅用于类型检查，避免循环导入
    from ..cases import PowerFlowProblem


@dataclass(frozen=True)
class PerturbationConfig:
    """负荷扰动的取值范围。

    全局缩放 ``sigma`` 模拟系统整体负荷水平，逐母线 ``eps`` 模拟
    节点间的不均衡。两者相乘，所以最坏情况下总负荷可以放大
    ``sigma_high * eps_p_high`` 倍——这个上界必须低于发电能力，
    由 ``load_capacity_sigma`` 把关。

    ``sigma_high`` 默认取 1.10 而不是蓝本示例里的 1.2，原因很实在：
    IEEE 39 的负荷裕度只有 ``sigma_bar = 1.178``（ΣPmax / ΣPd），
    直接取 1.2 会让最重场景连不计网损都无法平衡。而逐母线 ``eps``
    在 21 个负荷点上取平均后仍有约 2.2% 的标准差，所以 ``sigma_high``
    要留出约 6.5% 的余量（3σ），1.10 正好落在安全侧。
    想逼近失稳边界是**下一阶段**（病态样本）的课题，那时应当显式放宽。
    """

    sigma_low: float = 0.8
    sigma_high: float = 1.10
    eps_p_low: float = 0.9
    eps_p_high: float = 1.1
    eps_q_low: float = 0.9
    eps_q_high: float = 1.1

    def __post_init__(self) -> None:
        if self.sigma_low <= 0 or self.sigma_high <= 0:
            raise ValueError("sigma 必须为正")
        if self.sigma_low > self.sigma_high:
            raise ValueError("sigma_low 不能大于 sigma_high")
        for low, high, name in (
            (self.eps_p_low, self.eps_p_high, "eps_p"),
            (self.eps_q_low, self.eps_q_high, "eps_q"),
        ):
            if low <= 0 or high <= 0:
                raise ValueError(f"{name} 必须为正")
            if low > high:
                raise ValueError(f"{name}_low 不能大于 {name}_high")


@dataclass(frozen=True)
class LoadScenario:
    """一条采样出来的负荷场景。"""

    sample_id: int
    sigma: float
    pd_pu: np.ndarray
    qd_pu: np.ndarray
    eps_p: np.ndarray
    eps_q: np.ndarray

    @property
    def total_load_pu(self) -> float:
        return float(self.pd_pu.sum())


def base_load_pu(problem: "PowerFlowProblem") -> tuple[np.ndarray, np.ndarray]:
    """基准算例的母线负荷，单位标幺。"""

    bus = problem.bus
    return (
        bus[:, PD].astype(np.float64) / problem.base_mva,
        bus[:, QD].astype(np.float64) / problem.base_mva,
    )


def q_scaled_mask(problem: "PowerFlowProblem") -> np.ndarray:
    """标记哪些母线的无功会随负荷缩放。

    母线 9、24 原始 ``Qd`` 为负，代表无功补偿电容器而不是负荷。
    电容器是固定设备，系统负荷涨 20% 不会让它的容量跟着涨，
    所以这些母线的 Q 保持原值。
    """

    _, qd_pu = base_load_pu(problem)
    return qd_pu >= 0


def load_capacity_sigma(problem: "PowerFlowProblem") -> float:
    """发电有功上限与基准总负荷之比。

    这是数据构造蓝本里的 ``sigma_bar``：负荷缩放超过它，连不计网损的
    理想情况都无法平衡，程序性不可行。
    """

    from pypower.idx_gen import GEN_STATUS, PMAX

    gen = problem.gen
    online = gen[:, GEN_STATUS] > 0
    total_pmax_mw = float(gen[online, PMAX].sum())
    pd_pu, _ = base_load_pu(problem)
    total_load_mw = float(pd_pu.sum()) * problem.base_mva
    if total_load_mw <= 0:
        raise ValueError("基准总负荷为零，无法定义负荷裕度")
    return total_pmax_mw / total_load_mw


def sample_load_scenario(
    problem: "PowerFlowProblem",
    config: PerturbationConfig,
    sample_id: int,
    master_seed: int,
) -> LoadScenario:
    """按配置采样一条负荷场景，并施加护栏。

    违反护栏时**抛异常**，不静默截断——静默截断会让数据集悄悄偏向
    可行域内部，掩盖掉边界附近的样本。
    """

    sigma_bar = load_capacity_sigma(problem)
    if config.sigma_high > sigma_bar:
        raise ValueError(
            f"sigma_high={config.sigma_high:.4f} 超过系统负荷裕度 "
            f"sigma_bar={sigma_bar:.4f}，最重场景连不计网损都无法平衡"
        )

    rng = np.random.default_rng(np.random.SeedSequence([master_seed, sample_id]))
    pd_base, qd_base = base_load_pu(problem)
    nb = problem.nb

    sigma = float(rng.uniform(config.sigma_low, config.sigma_high))
    eps_p = rng.uniform(config.eps_p_low, config.eps_p_high, size=nb)
    eps_q = rng.uniform(config.eps_q_low, config.eps_q_high, size=nb)

    pd_pu = sigma * eps_p * pd_base
    # 负 Qd（电容器）不缩放，保持原值。
    scaled_q = sigma * eps_q * qd_base
    qd_pu = np.where(q_scaled_mask(problem), scaled_q, qd_base)

    total_load_pu = float(pd_pu.sum())
    limit_pu = sigma_bar * float(pd_base.sum())
    if total_load_pu > limit_pu:
        raise ValueError(
            f"样本 {sample_id} 总负荷 {total_load_pu:.6f} pu 超过上限 "
            f"{limit_pu:.6f} pu"
        )

    return LoadScenario(
        sample_id=sample_id,
        sigma=sigma,
        pd_pu=pd_pu,
        qd_pu=qd_pu,
        eps_p=eps_p,
        eps_q=eps_q,
    )


def apply_scenario(
    base: "PowerFlowProblem",
    scenario: LoadScenario,
) -> "PowerFlowProblem":
    """把负荷场景套到基准算例上，返回新的潮流问题。

    ``PowerFlowProblem`` 是冻结的，但其内部的 ``ppci`` 字典不是，
    所以必须先深拷贝再改。``Sbus`` 必须跟着一起重算，否则
    ``build_external_result`` / ``pfsoln`` 会拿着旧注入回算，
    得到的发电机出力对不上。

    ``ppc``（外部编号）与 ``ppci``（内部编号）描述的是同一个系统，
    必须**一起更新**，否则读 ``problem.ppc`` 的人会拿到过期的负荷。
    两者的行序由 ``ppci["order"]["bus"]`` 对应。
    """

    ppc = copy.deepcopy(base.ppc)
    ppci = copy.deepcopy(base.ppci)
    rows = external_bus_rows(base)
    ppc["bus"][rows, PD] = scenario.pd_pu * base.base_mva
    ppc["bus"][rows, QD] = scenario.qd_pu * base.base_mva
    ppci["bus"][:, PD] = scenario.pd_pu * base.base_mva
    ppci["bus"][:, QD] = scenario.qd_pu * base.base_mva
    sbus = makeSbus(base.base_mva, ppci["bus"], ppci["gen"])
    # Ybus/Yf/Yt/ref/pv/pq 与负荷无关，直接复用；V0 保持平启动。
    return replace(base, ppc=ppc, ppci=ppci, Sbus=sbus)
