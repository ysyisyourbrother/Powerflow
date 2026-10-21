"""赛事对齐格式的回归测试：字段、掩码、标签、落盘。"""

import numpy as np
import pytest
from pypower.idx_brch import BR_B, F_BUS, PF, PT, QF, QT, T_BUS
from pypower.idx_bus import BUS_TYPE

from powerflow import build_external_result, prepare_case39, solve_newton_raphson
from powerflow.dataset import (
    CompetitionDataset,
    MaskConfig,
    competition_topology,
    label_competition_sample,
    write_competition_dataset,
)
from powerflow.dataset.competition import (
    BRANCH_TYPE_LINE,
    BRANCH_TYPE_TRANSFORMER,
    MASK_KNOWN,
    MASK_PREDICT,
    SAMPLE_SCALAR_FIELDS,
    structural_masks,
)
from powerflow.dataset.sampler import (
    PerturbationConfig,
    apply_scenario,
    base_load_pu,
    sample_load_scenario,
)

MASTER_SEED = 20261019
MASK_SEED = 7


@pytest.fixture(scope="module")
def base():
    return prepare_case39(flat=True)


def _neutral_problem(base):
    """sigma=1, eps=1 的基准运行方式。"""

    pd_pu, qd_pu = base_load_pu(base)
    from powerflow.dataset.sampler import LoadScenario

    scenario = LoadScenario(
        sample_id=0,
        sigma=1.0,
        pd_pu=pd_pu,
        qd_pu=qd_pu,
        eps_p=np.ones(base.nb),
        eps_q=np.ones(base.nb),
    )
    return apply_scenario(base, scenario)


def _sample(base, *, sample_id=0, mode="structural"):
    problem = apply_scenario(
        base,
        sample_load_scenario(base, PerturbationConfig(), sample_id, MASTER_SEED),
    )
    return problem, label_competition_sample(
        problem,
        sample_id=sample_id,
        mask_config=MaskConfig(mode=mode),
        master_seed=MASTER_SEED,
        mask_seed=MASK_SEED,
    )


def test_topology_uses_b_over_two_and_branch_types(base):
    topology = competition_topology(base)

    assert topology["nb"] == 39
    assert topology["ne"] == 46
    assert topology["edge_index"].shape == (2, 46)
    assert topology["branch_features"].shape == (46, 3)
    assert topology["node_ids"].tolist() == list(range(1, 40))

    # 赛题要 B/2。这里用原始 BR_B 反推，防止有人把 B 当 B/2 写出去。
    assert np.allclose(topology["branch_features"][:, 2] * 2.0, base.branch[:, BR_B])

    branch_type = topology["branch_type"]
    assert int((branch_type == BRANCH_TYPE_LINE).sum()) == 34
    assert int((branch_type == BRANCH_TYPE_TRANSFORMER).sum()) == 12
    # 首末端电压等级来自母线 BASE_KV，case39 全部是 345 kV。
    assert np.allclose(topology["node_kv"], 345.0)
    assert np.allclose(topology["from_kv"], 345.0)
    assert np.allclose(topology["to_kv"], 345.0)


def test_structural_masks_have_standard_power_flow_counts(base):
    problem = _neutral_problem(base)
    sample = label_competition_sample(
        problem,
        sample_id=0,
        mask_config=MaskConfig(mode="structural"),
    )

    # 39 节点：29 PQ + 9 PV + 1 REF。
    # P 已知 29+9=38；Q 已知 29；V 已知 9+1=10。
    assert sample.n_known_p == 38
    assert sample.n_known_q == 29
    assert sample.n_known_v == 10

    bus_type = problem.bus[:, BUS_TYPE]
    p_mask, q_mask, v_mask = structural_masks(bus_type)
    assert np.array_equal(sample.node_p_mask, p_mask)
    assert np.array_equal(sample.node_q_mask, q_mask)
    assert np.array_equal(sample.node_v_mask, v_mask)

    # 被掩码的位置一律填 0，作为占位。
    assert np.all(sample.node_p_mw[sample.node_p_mask == MASK_PREDICT] == 0.0)
    assert np.all(sample.node_q_mvar[sample.node_q_mask == MASK_PREDICT] == 0.0)
    assert np.all(sample.node_vm_pu[sample.node_v_mask == MASK_PREDICT] == 0.0)


def test_known_inputs_match_true_injections(base):
    problem = _neutral_problem(base)
    sample = label_competition_sample(
        problem,
        sample_id=0,
        mask_config=MaskConfig(mode="structural"),
    )

    # PQ/PV 的已知 P 就是计划净注入，和潮流解只差求解容差。
    known_p = sample.node_p_mask == MASK_KNOWN
    known_q = sample.node_q_mask == MASK_KNOWN
    known_v = sample.node_v_mask == MASK_KNOWN
    assert np.max(np.abs(sample.node_p_mw[known_p] - sample.aux_net_p_mw[known_p])) < 1e-6
    assert np.max(np.abs(sample.node_q_mvar[known_q] - sample.aux_net_q_mvar[known_q])) < 1e-6
    # PV/REF 的已知 V 是设定值，和收敛解一致。
    assert np.max(np.abs(sample.node_vm_pu[known_v] - sample.target_vm_pu[known_v])) < 1e-8


def test_branch_targets_match_pypower(base):
    problem, sample = _sample(base, sample_id=3)
    result = solve_newton_raphson(problem, tol=1e-10, max_iter=40)
    external = build_external_result(problem, result)
    branch = external["branch"]

    assert np.max(np.abs(sample.target_pf_mw - branch[:, PF])) < 1e-8
    assert np.max(np.abs(sample.target_pt_mw - branch[:, PT])) < 1e-8
    assert np.max(np.abs(sample.target_qf_mvar - branch[:, QF])) < 1e-8
    assert np.max(np.abs(sample.target_qt_mvar - branch[:, QT])) < 1e-8

    f = np.asarray(problem.branch[:, F_BUS], dtype=int)
    t = np.asarray(problem.branch[:, T_BUS], dtype=int)
    expected = np.degrees(
        np.angle(result.voltage)[f] - np.angle(result.voltage)[t]
    )
    assert np.max(np.abs(sample.target_branch_dtheta_deg - expected)) < 1e-12


def test_energy_balance_between_branch_targets_and_loss(base):
    _, sample = _sample(base, sample_id=5)
    branch_loss = float((sample.target_pf_mw + sample.target_pt_mw).sum())
    assert abs(branch_loss - sample.loss_mw) < 1e-6


def test_mask_modes_are_valid(base):
    problem = apply_scenario(
        base,
        sample_load_scenario(base, PerturbationConfig(), 9, MASTER_SEED),
    )

    for mode in ("structural", "dropout", "extra", "mixed"):
        sample = label_competition_sample(
            problem,
            sample_id=9,
            mask_config=MaskConfig(mode=mode, dropout_prob=0.2, extra_prob=0.2),
            master_seed=MASTER_SEED,
            mask_seed=MASK_SEED,
        )
        for mask in (sample.node_p_mask, sample.node_q_mask, sample.node_v_mask):
            assert set(np.unique(mask).tolist()) <= {0, 1}
        assert 0 <= sample.n_known_p <= 39
        assert 0 <= sample.n_known_q <= 39
        assert 0 <= sample.n_known_v <= 39


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    base = prepare_case39(flat=True)
    root = tmp_path_factory.mktemp("competition")
    samples = []
    for sample_id in range(5):
        problem = apply_scenario(
            base,
            sample_load_scenario(base, PerturbationConfig(), sample_id, MASTER_SEED),
        )
        samples.append(
            label_competition_sample(
                problem,
                sample_id=sample_id,
                mask_config=MaskConfig(mode="mixed"),
                master_seed=MASTER_SEED,
                mask_seed=MASK_SEED,
            )
        )
    manifest = write_competition_dataset(
        root,
        competition_topology(base),
        samples,
        config={"master_seed": MASTER_SEED, "n_samples": 5},
        shard_size=2,
        topology_name="case39_base",
    )
    return root, samples, manifest


def test_round_trip_preserves_every_sample_field(built):
    root, samples, manifest = built
    dataset = CompetitionDataset(root)

    assert len(dataset) == 5
    assert [entry["count"] for entry in manifest["shards"]] == [2, 2, 1]

    for index, original in enumerate(samples):
        item = dataset[index]

        # 输入：P/Q/V 三列 + 三通道掩码。
        assert np.array_equal(item["node_features"][:, 0], original.node_p_mw)
        assert np.array_equal(item["node_features"][:, 1], original.node_q_mvar)
        assert np.array_equal(item["node_features"][:, 2], original.node_vm_pu)
        assert np.array_equal(item["node_masks"][:, 0], original.node_p_mask)
        assert np.array_equal(item["node_masks"][:, 1], original.node_q_mask)
        assert np.array_equal(item["node_masks"][:, 2], original.node_v_mask)

        # 官方标签。
        assert np.array_equal(item["targets"]["vm_pu"], original.target_vm_pu)
        assert np.array_equal(
            item["targets"]["dtheta_deg"], original.target_branch_dtheta_deg
        )
        assert np.array_equal(item["targets"]["pf_mw"], original.target_pf_mw)
        assert np.array_equal(item["targets"]["pt_mw"], original.target_pt_mw)
        assert np.array_equal(item["targets"]["qf_mvar"], original.target_qf_mvar)
        assert np.array_equal(item["targets"]["qt_mvar"], original.target_qt_mvar)

        # 辅助标签。
        assert np.array_equal(item["aux"]["va_deg"], original.aux_va_deg)
        assert np.array_equal(item["aux"]["net_p_mw"], original.aux_net_p_mw)
        assert np.array_equal(item["aux"]["net_q_mvar"], original.aux_net_q_mvar)

        # 标量元数据。
        for field in SAMPLE_SCALAR_FIELDS:
            assert item["meta"][field] == getattr(original, field), field


def test_reader_shapes_and_batch(built):
    root, _, _ = built
    dataset = CompetitionDataset(root)
    item = dataset[0]

    assert item["node_features"].shape == (39, 3)
    assert item["node_masks"].shape == (39, 3)
    assert item["edge_index"].shape == (2, 46)
    assert item["branch_features"].shape == (46, 3)
    assert item["targets"]["vm_pu"].shape == (39,)
    assert item["targets"]["dtheta_deg"].shape == (46,)
    assert item["targets"]["pf_mw"].shape == (46,)
    assert item["node_ids"].shape == (39,)
    assert item["branch_ids"].shape == (46,)

    batch = next(dataset.iter_batches(batch_size=3))
    assert batch["node_features"].shape == (3, 39, 3)
    assert batch["node_masks"].shape == (3, 39, 3)
    assert batch["targets"]["vm_pu"].shape == (3, 39)
    assert batch["targets"]["pf_mw"].shape == (3, 46)
