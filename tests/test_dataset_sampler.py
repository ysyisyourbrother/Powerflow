"""采样器测试：确定性、保号约束、护栏。"""

import numpy as np
import pytest
from pypower.idx_bus import PD, QD

from powerflow import prepare_case39
from powerflow.dataset import (
    PerturbationConfig,
    apply_scenario,
    base_load_pu,
    load_capacity_sigma,
    q_scaled_mask,
    sample_load_scenario,
)

MASTER_SEED = 20261014


@pytest.fixture(scope="module")
def base():
    return prepare_case39(flat=True)


def test_sampler_is_a_pure_function_of_seed_and_id(base):
    """同 (master_seed, sample_id) 必须逐位复现；换 id 必须不同。"""

    config = PerturbationConfig()
    first = sample_load_scenario(base, config, 3, MASTER_SEED)
    again = sample_load_scenario(base, config, 3, MASTER_SEED)
    other = sample_load_scenario(base, config, 4, MASTER_SEED)

    assert first.sigma == again.sigma
    assert np.array_equal(first.pd_pu, again.pd_pu)
    assert np.array_equal(first.qd_pu, again.qd_pu)
    assert np.array_equal(first.eps_p, again.eps_p)
    assert np.array_equal(first.eps_q, again.eps_q)

    assert not np.array_equal(first.pd_pu, other.pd_pu)

    # 随机访问：不重放前序样本也能拿到同一条。
    assert np.array_equal(
        sample_load_scenario(base, config, 3, MASTER_SEED).pd_pu, first.pd_pu
    )


def test_sampler_preserves_capacitors_and_topology(base):
    """只有负荷在变，拓扑与电容器必须原封不动。"""

    scenario = sample_load_scenario(base, PerturbationConfig(), 11, MASTER_SEED)
    problem = apply_scenario(base, scenario)
    _, qd_base = base_load_pu(problem)
    mask = q_scaled_mask(problem)
    negatives = np.flatnonzero(~mask)

    assert (negatives + 1).tolist() == [9, 24]
    # 负 Qd 是电容器：容量不随系统负荷涨落。
    assert np.array_equal(scenario.qd_pu[negatives], qd_base[negatives])
    assert np.all(scenario.qd_pu[negatives] < 0)

    # 这些母线的无功列也确实没被写进去。
    assert np.array_equal(problem.ppci["bus"][negatives, QD], base.ppci["bus"][negatives, QD])

    # 拓扑相关的一切都与 base 逐位相同（同一对象级复用）。
    assert np.array_equal(problem.bus[:, 1], base.bus[:, 1])  # BUS_TYPE
    assert problem.ref is base.ref
    assert problem.pv is base.pv
    assert problem.pq is base.pq
    assert problem.Ybus is base.Ybus
    assert np.array_equal(problem.V0, base.V0)

    # 负荷下标不一致的地方确实变了。
    assert not np.array_equal(problem.ppci["bus"][:, PD], base.ppci["bus"][:, PD])


def test_apply_scenario_keeps_ppc_and_ppci_in_sync(base):
    """两份数据（外部编号 / 内部编号）必须一起更新。

    只改 ppci 会让读 problem.ppc 的人拿到过期负荷——这是静默错误，
    所以显式测。
    """

    from powerflow import external_bus_rows
    from pypower.idx_bus import PD as PD_IDX

    scenario = sample_load_scenario(base, PerturbationConfig(), 5, MASTER_SEED)
    problem = apply_scenario(base, scenario)
    rows = external_bus_rows(base)

    assert np.allclose(problem.ppc["bus"][rows, PD_IDX], scenario.pd_pu * base.base_mva)
    assert np.allclose(problem.ppc["bus"][rows, QD], scenario.qd_pu * base.base_mva)
    assert np.allclose(problem.ppci["bus"][:, PD], scenario.pd_pu * base.base_mva)
    assert np.allclose(problem.ppci["bus"][:, QD], scenario.qd_pu * base.base_mva)

    # Sbus 必须跟着负荷重算。发电计划不变，所以净注入的实部变化
    # 应当恰好等于负荷变化的相反数——拿 base 对比，不是自比自。
    pd_base, qd_base = base_load_pu(base)
    assert np.allclose(problem.Sbus.real, base.Sbus.real - (scenario.pd_pu - pd_base))
    assert np.allclose(problem.Sbus.imag, base.Sbus.imag - (scenario.qd_pu - qd_base))
    # 没重算 Sbus 就会退化成这一条（只改表、不改注入）。
    assert not np.allclose(problem.Sbus.real, base.Sbus.real)

    # base 本身没被改动。
    assert not np.array_equal(base.ppci["bus"][:, PD], problem.ppci["bus"][:, PD])


def test_guardrail_holds_over_many_draws(base):
    config = PerturbationConfig()
    sigma_bar = load_capacity_sigma(base)
    assert config.sigma_high <= sigma_bar

    pd_base, _ = base_load_pu(base)
    limit = sigma_bar * float(pd_base.sum())

    for sample_id in range(1000):
        scenario = sample_load_scenario(base, config, sample_id, MASTER_SEED)
        assert np.all(np.isfinite(scenario.pd_pu))
        assert np.all(np.isfinite(scenario.qd_pu))
        assert np.all(scenario.pd_pu >= 0)
        assert config.sigma_low <= scenario.sigma <= config.sigma_high
        assert scenario.total_load_pu <= limit


def test_guardrail_rejects_impossibly_high_sigma(base):
    """sigma_high 超过系统负荷裕度时必须抛异常，而不是静默截断。"""

    sigma_bar = load_capacity_sigma(base)
    with pytest.raises(ValueError, match="sigma_high"):
        sample_load_scenario(
            base,
            PerturbationConfig(sigma_high=sigma_bar + 0.05),
            0,
            MASTER_SEED,
        )


def test_config_validates_its_own_ranges():
    with pytest.raises(ValueError, match="sigma_low"):
        PerturbationConfig(sigma_low=1.3, sigma_high=1.1)
    with pytest.raises(ValueError, match="eps_p"):
        PerturbationConfig(eps_p_low=-1.0)
    with pytest.raises(ValueError, match="eps_q"):
        PerturbationConfig(eps_q_low=1.5, eps_q_high=1.1)


def test_load_capacity_sigma_matches_sum_pmax_over_sum_pd(base):
    from pypower.idx_gen import GEN_STATUS, PMAX

    online = base.gen[:, GEN_STATUS] > 0
    expected = float(base.gen[online, PMAX].sum()) / (base.ppci["bus"][:, PD].sum())
    assert abs(load_capacity_sigma(base) - expected) < 1e-12
