"""算例准备层的回归测试：委托重构、平启动、基准复现、结构。"""

import numpy as np
from pypower.idx_bus import BUS_I, BUS_TYPE, VA, VM
from pypower.idx_gen import GEN_BUS, GEN_STATUS, VG

from powerflow import (
    external_bus_rows,
    flat_start,
    prepare_case39,
    prepare_case9,
    solve_newton_raphson,
)
from powerflow.dataset import (
    LoadScenario,
    MaskConfig,
    apply_scenario,
    base_load_pu,
    label_competition_sample,
    q_scaled_mask,
)

# 参照值取自 dataset/ieee39/reference/summary.json。
REFERENCE_VM_MIN = 0.9820
REFERENCE_VM_MAX = 1.0636
REFERENCE_VA_MIN_DEG = -14.5353
REFERENCE_VA_MAX_DEG = 4.4684
REFERENCE_LOSS_MW = 43.64112576081334
REFERENCE_LOAD_MW = 6254.23


def _delta_scenario(base):
    """构造 sigma=1, eps=1 的「零扰动」场景，用来复现基准算例。"""

    pd_pu, qd_pu = base_load_pu(base)
    return LoadScenario(
        sample_id=0,
        sigma=1.0,
        pd_pu=pd_pu,
        qd_pu=qd_pu,
        eps_p=np.ones(base.nb),
        eps_q=np.ones(base.nb),
    )


def test_prepare_case9_unchanged_after_refactor():
    """prepare_case9 现在委托给通用的 prepare_case，行为必须逐位不变。"""

    problem = prepare_case9()
    result = solve_newton_raphson(problem, tol=1e-10)
    assert result.converged
    assert result.iterations == 4
    assert result.final_norm < 1e-10


def test_case39_flat_start_is_not_the_bundled_solution():
    """case39 自带的 VM/VA 本身就是一组解，必须被 flat_start 抹掉。

    若误用那组解当启动点，牛顿法 1 次迭代就收敛，iterations 元数据
    会彻底失去意义。这里的断言是防回归的闸门。
    """

    problem = prepare_case39(flat=True)
    bus_type = problem.bus[:, BUS_TYPE]

    # PQ 母线平启动为 1.0，PV 与平衡母线取发电机 VG 设定。
    vg_set = np.zeros(problem.nb)
    for gen_index in np.flatnonzero(problem.gen[:, GEN_STATUS] > 0):
        vg_set[int(problem.gen[gen_index, GEN_BUS])] = problem.gen[gen_index, VG]
    expected = np.where(bus_type == 1, 1.0, vg_set)

    assert np.allclose(np.abs(problem.V0), expected, atol=1e-12)
    assert np.allclose(np.angle(problem.V0), 0.0, atol=1e-12)

    result = solve_newton_raphson(problem, tol=1e-10)
    assert result.converged
    assert result.iterations >= 3


def test_flat_start_clears_voltage_columns():
    from pypower.api import case39

    ppc = flat_start(case39())
    assert np.all(ppc["bus"][:, VM] == 1.0)
    assert np.all(ppc["bus"][:, VA] == 0.0)


def test_case39_reproduces_reference():
    """sigma=1, eps=1 必须复现 dataset/ieee39/reference 里那组数。"""

    base = prepare_case39(flat=True)
    problem = apply_scenario(base, _delta_scenario(base))
    sample = label_competition_sample(
        problem,
        sample_id=0,
        sigma=1.0,
        mask_config=MaskConfig(mode="structural"),
    )

    assert sample.converged
    assert abs(sample.target_vm_pu.min() - REFERENCE_VM_MIN) < 1e-3
    assert abs(sample.target_vm_pu.max() - REFERENCE_VM_MAX) < 1e-3

    va_deg = sample.aux_va_deg
    assert abs(va_deg.min() - REFERENCE_VA_MIN_DEG) < 1e-3
    assert abs(va_deg.max() - REFERENCE_VA_MAX_DEG) < 1e-3

    assert abs(sample.loss_mw - REFERENCE_LOSS_MW) < 0.01
    assert abs(sample.total_load_mw - REFERENCE_LOAD_MW) < 0.05


def test_case39_structure():
    problem = prepare_case39(flat=True)
    assert problem.nb == 39
    assert problem.branch.shape[0] == 46
    assert len(problem.pv) == 9
    assert len(problem.pq) == 29
    assert len(problem.ref) == 1

    # 牛顿法状态维数：非平衡相角 + PQ 电压幅值。
    assert len(problem.pv) + 2 * len(problem.pq) == 67


def test_external_bus_rows_is_a_bijection():
    """内部行 → 外部行的映射必须是双射，否则 ppc 会被写错行。"""

    problem = prepare_case39(flat=True)
    rows = external_bus_rows(problem)

    assert rows.shape == (problem.nb,)
    assert len(set(rows.tolist())) == problem.nb
    assert set(rows.tolist()) == set(range(problem.nb))

    # 逐内部行核对：映射过去的行，其外部母线编号必须等于 i2e[i]。
    i2e = np.asarray(problem.ppci["order"]["bus"]["i2e"], dtype=int)
    external_ids = problem.ppc["bus"][rows, BUS_I].astype(int)
    assert np.array_equal(external_ids, i2e)


def test_case39_capacitor_buses_have_negative_qd():
    """母线 9、24 的负 Qd 是无功补偿，不是负荷——必须保持负号。"""

    problem = prepare_case39(flat=True)
    _, qd_pu = base_load_pu(problem)
    mask = q_scaled_mask(problem)

    negative = np.flatnonzero(~mask)
    assert (negative + 1).tolist() == [9, 24]
    assert np.all(qd_pu[negative] < 0)
