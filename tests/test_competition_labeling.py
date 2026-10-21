"""赛事格式打标测试：跨实现一致、元数据、物理自洽、不收敛样本保留。"""

import numpy as np
import pytest

from powerflow import prepare_case39
from powerflow.dataset import (
    LoadScenario,
    MaskConfig,
    PerturbationConfig,
    apply_scenario,
    assert_flat_start,
    base_load_pu,
    cross_check_pypower,
    label_competition_sample,
    sample_load_scenario,
    topology_hash,
)
from powerflow.dataset.competition import (
    SAMPLE_ARRAY_FIELDS,
    SAMPLE_SCALAR_FIELDS,
    expected_flat_voltage,
)

MASTER_SEED = 20261014
MASK_SEED = 7


@pytest.fixture(scope="module")
def base():
    return prepare_case39(flat=True)


def _problem(base, sample_id):
    scenario = sample_load_scenario(base, PerturbationConfig(), sample_id, MASTER_SEED)
    return apply_scenario(base, scenario), scenario


def _sample(problem, sample_id, sigma=1.0):
    return label_competition_sample(
        problem,
        sample_id=sample_id,
        sigma=sigma,
        mask_config=MaskConfig(mode="structural"),
        mask_seed=MASK_SEED,
    )


def test_custom_solver_matches_pypower(base):
    """在同一个扰动后、平启动的算例上，自研解与 PYPOWER 必须一致。"""

    for sample_id in (0, 17, 42):
        problem, scenario = _problem(base, sample_id)
        sample = _sample(problem, sample_id, scenario.sigma)
        assert sample.converged

        voltage = sample.target_vm_pu * np.exp(1j * np.deg2rad(sample.aux_va_deg))
        report = cross_check_pypower(problem, voltage)
        assert report["pypower_converged"]
        assert report["max_dvm_pu"] < 1e-8
        assert report["max_dva_rad"] < 1e-6
        assert abs(report["d_loss_mw"]) < 1e-6


def test_metadata_is_populated(base):
    problem, scenario = _problem(base, 3)
    sample = _sample(problem, 3, scenario.sigma)

    assert sample.converged
    assert sample.final_norm < 1e-10
    assert sample.initial_norm > 1.0, "平启动的初始失配不该是小量"
    assert sample.iterations >= 1
    assert sample.sample_id == 3
    assert abs(sample.sigma - scenario.sigma) < 1e-15
    assert len(sample.topology_hash) == 16
    assert len(sample.case_hash) == 16
    assert abs(sample.total_load_mw - scenario.total_load_pu * base.base_mva) < 1e-9


def test_labels_satisfy_power_flow_equations(base):
    """不信任收敛标志位，用 Ybus 独立复算残差。"""

    problem, scenario = _problem(base, 9)
    sample = _sample(problem, 9, scenario.sigma)

    voltage = sample.target_vm_pu * np.exp(1j * np.deg2rad(sample.aux_va_deg))
    net_injection = voltage * np.conj(problem.Ybus.dot(voltage))

    for bus_index in problem.pv:
        assert abs(net_injection[bus_index].real - problem.Sbus[bus_index].real) < 1e-9
    for bus_index in problem.pq:
        assert abs(net_injection[bus_index] - problem.Sbus[bus_index]) < 1e-9

    # 辅助量就是这次复算得到的真实净注入。
    assert np.allclose(net_injection.real * base.base_mva, sample.aux_net_p_mw, atol=1e-9)
    assert np.allclose(net_injection.imag * base.base_mva, sample.aux_net_q_mvar, atol=1e-9)

    # 平衡母线相角归零。
    slack = problem.ref[0]
    assert abs(sample.aux_va_deg[slack]) < 1e-12


def test_nonconvergent_sample_is_kept_not_raised(base):
    """不收敛样本必须照样返回完整样本，而不是抛异常或被丢掉。"""

    # 绕过护栏直接构造一个远超可行域的负荷场景（8 倍负荷）。
    pd_base, qd_base = base_load_pu(base)
    extreme = LoadScenario(
        sample_id=-1,
        sigma=8.0,
        pd_pu=8.0 * pd_base,
        qd_pu=8.0 * qd_base,
        eps_p=np.ones(base.nb),
        eps_q=np.ones(base.nb),
    )
    problem = apply_scenario(base, extreme)
    # 掩码随机种子要求非负样本号；这里只测打标，不关心编号是否连续。
    sample = _sample(problem, 0, 8.0)

    assert not sample.converged
    assert sample.iterations == 40
    assert sample.target_vm_pu.shape == (base.nb,)
    assert sample.target_branch_dtheta_deg.shape == (base.branch.shape[0],)
    for field in SAMPLE_ARRAY_FIELDS + SAMPLE_SCALAR_FIELDS:
        assert getattr(sample, field) is not None


def test_assert_flat_start_rejects_solved_state():
    """用算例自带的那组解当启动点必须被断言拦下。"""

    solved = prepare_case39(flat=False)
    with pytest.raises(ValueError, match="平启动"):
        assert_flat_start(solved)

    flat = prepare_case39(flat=True)
    assert_flat_start(flat)
    assert np.allclose(np.abs(flat.V0), expected_flat_voltage(flat))


def test_topology_hash_is_shared_across_load_scenarios(base):
    """同一张网架的所有负荷场景必须共享 topology_hash。"""

    hashes = {topology_hash(_problem(base, i)[0]) for i in range(6)}
    assert len(hashes) == 1


def test_case_hash_detects_distinct_inputs(base):
    hashes = {_sample(_problem(base, i)[0], i).case_hash for i in range(6)}
    assert len(hashes) == 6
