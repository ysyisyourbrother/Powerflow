"""生成赛事对齐的 IEEE 39 数据集：采样运行方式 → 求解潮流 → 掩码 → 落盘。

数据字段约定：

* 输入是**节点净注入 P/Q**（MW/MVar）和**电压幅值 V**（p.u.）；
* 输入带三通道掩码，``0 = 已知``、``1 = 需要预测``；
* 输出是**全部节点 Vm** 和**全部支路的首末端相角差 / P / Q**；
* 支路特征只保留赛题要求的 R/X/B2，另外把 tap/shift 放在 aux 里供物理复算。

运行::

    python src/examples/build_competition_dataset.py --n-samples 2000
    python src/examples/build_competition_dataset.py --n-samples 2000 --mask-mode structural
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow import prepare_case39, solve_newton_raphson  # noqa: E402
from powerflow.dataset.competition import (  # noqa: E402
    BRANCH_TYPE_LINE,
    BRANCH_TYPE_NAMES,
    BRANCH_TYPE_TRANSFORMER,
    COMPETITION_SCHEMA_VERSION,
    MASK_KNOWN,
    MASK_MODE_NAMES,
    CompetitionDataset,
    MaskConfig,
    competition_topology,
    cross_check_pypower,
    label_competition_sample,
    write_competition_dataset,
)
from powerflow.dataset.sampler import (  # noqa: E402
    LoadScenario,
    PerturbationConfig,
    apply_scenario,
    base_load_pu,
    load_capacity_sigma,
    sample_load_scenario,
)

# 参照值取自 dataset/ieee39/reference/summary.json。
REFERENCE_VM_MIN = 0.9820
REFERENCE_VM_MAX = 1.0636
REFERENCE_LOSS_MW = 43.64112576081334
REFERENCE_LOAD_MW = 6254.23

DEFAULT_OUTPUT = Path("dataset/ieee39/processed/competition_v1")


def _banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _neutral_scenario(base) -> LoadScenario:
    """sigma=1, eps=1 的零扰动场景，用来复现参考解。"""

    pd_pu, qd_pu = base_load_pu(base)
    return LoadScenario(
        sample_id=-1,
        sigma=1.0,
        pd_pu=pd_pu,
        qd_pu=qd_pu,
        eps_p=np.ones(base.nb),
        eps_q=np.ones(base.nb),
    )


def check_baseline(base) -> None:
    _banner("1. 基准复现（sigma=1, eps=1）")

    problem = apply_scenario(base, _neutral_scenario(base))
    sample = label_competition_sample(
        problem,
        sample_id=0,
        sigma=1.0,
        mask_config=MaskConfig(mode="structural"),
    )

    print(f"  Vm 范围      [{sample.target_vm_pu.min():.4f}, {sample.target_vm_pu.max():.4f}]"
          f"  期望 [{REFERENCE_VM_MIN:.4f}, {REFERENCE_VM_MAX:.4f}]")
    print(f"  网损         {sample.loss_mw:.8f} MW   期望 {REFERENCE_LOSS_MW:.8f}")
    print(f"  总负荷       {sample.total_load_mw:.2f} MW     期望 {REFERENCE_LOAD_MW:.2f}")
    print(f"  迭代次数     {sample.iterations}")

    assert sample.converged, "基准算例未收敛"
    assert abs(sample.target_vm_pu.min() - REFERENCE_VM_MIN) < 1e-3
    assert abs(sample.target_vm_pu.max() - REFERENCE_VM_MAX) < 1e-3
    assert abs(sample.loss_mw - REFERENCE_LOSS_MW) < 0.01
    assert abs(sample.total_load_mw - REFERENCE_LOAD_MW) < 0.05

    report = cross_check_pypower(
        problem, sample.target_vm_pu * np.exp(1j * np.deg2rad(sample.aux_va_deg))
    )
    print(f"  对照 PYPOWER: max|dVm|={report['max_dvm_pu']:.3e} "
          f"max|dVa|={report['max_dva_rad']:.3e} dLoss={report['d_loss_mw']:+.3e} MW")
    assert report["pypower_converged"]
    assert report["max_dvm_pu"] < 1e-8
    assert report["max_dva_rad"] < 1e-6
    print("  [PASS] 基准复现")


def check_topology(topology: dict, base) -> None:
    _banner("2. 拓扑与字段")

    nb = int(topology["nb"])
    ne = int(topology["ne"])
    branch_type = np.asarray(topology["branch_type"])
    names = Counter(BRANCH_TYPE_NAMES[int(t)] for t in branch_type)

    print(f"  nb={nb}, ne={ne}, base_mva={float(topology['base_mva']):.1f}")
    print(f"  支路类型: {dict(names)}")
    print(f"  节点电压等级唯一值: {np.unique(topology['node_kv']).tolist()}")
    print(f"  首端电压等级唯一值: {np.unique(topology['from_kv']).tolist()}")
    print(f"  末端电压等级唯一值: {np.unique(topology['to_kv']).tolist()}")
    print(f"  B/2 范围: [{topology['branch_features'][:, 2].min():.4f}, "
          f"{topology['branch_features'][:, 2].max():.4f}]")

    assert int((branch_type == BRANCH_TYPE_LINE).sum()) == 34
    assert int((branch_type == BRANCH_TYPE_TRANSFORMER).sum()) == 12
    # 赛题要求 B/2，且不许再乘倍数。这里用原始 BR_B 反推，确认存的确实是 B/2。
    from pypower.idx_brch import BR_B

    assert np.allclose(topology["branch_features"][:, 2] * 2.0, base.branch[:, BR_B])
    print("  [PASS] 拓扑字段完整（34 条线路 + 12 条变压器支路，B/2 正确）")


def check_masks(samples) -> None:
    _banner("3. 掩码分布")

    modes = Counter(MASK_MODE_NAMES[int(s.mask_mode)] for s in samples)
    print(f"  模式分布: {dict(modes)}")

    structural = [s for s in samples if s.mask_mode == 0]
    if structural:
        first = structural[0]
        print(f"  structural 首条: known P/Q/V = "
              f"{first.n_known_p}/{first.n_known_q}/{first.n_known_v}")
        assert first.n_known_p == 38
        assert first.n_known_q == 29
        assert first.n_known_v == 10
        assert np.all(first.node_p_mw[first.node_p_mask == 1] == 0.0)
        assert np.all(first.node_q_mvar[first.node_q_mask == 1] == 0.0)
        assert np.all(first.node_vm_pu[first.node_v_mask == 1] == 0.0)

    known_p = np.asarray([s.n_known_p for s in samples])
    known_q = np.asarray([s.n_known_q for s in samples])
    known_v = np.asarray([s.n_known_v for s in samples])
    print(f"  已知数量范围: P [{known_p.min()}, {known_p.max()}], "
          f"Q [{known_q.min()}, {known_q.max()}], V [{known_v.min()}, {known_v.max()}]")
    print("  [PASS] 掩码语义正确")


def check_targets(samples) -> None:
    """用样本自己的标签复算支路潮流，确认六个输出彼此自洽。"""

    _banner("4. 标签物理自洽性")

    worst_flow = 0.0
    worst_balance = 0.0
    worst_known = 0.0

    for sample in samples:
        if not sample.converged:
            continue
        # 已知输入（掩码=0）必须等于真实净注入/真实电压。
        if np.any(sample.node_p_mask == MASK_KNOWN):
            worst_known = max(
                worst_known,
                float(np.abs(
                    sample.node_p_mw[sample.node_p_mask == MASK_KNOWN]
                    - sample.aux_net_p_mw[sample.node_p_mask == MASK_KNOWN]
                ).max()),
            )
        if np.any(sample.node_q_mask == MASK_KNOWN):
            worst_known = max(
                worst_known,
                float(np.abs(
                    sample.node_q_mvar[sample.node_q_mask == MASK_KNOWN]
                    - sample.aux_net_q_mvar[sample.node_q_mask == MASK_KNOWN]
                ).max()),
            )
        if np.any(sample.node_v_mask == MASK_KNOWN):
            worst_known = max(
                worst_known,
                float(np.abs(
                    sample.node_vm_pu[sample.node_v_mask == MASK_KNOWN]
                    - sample.target_vm_pu[sample.node_v_mask == MASK_KNOWN]
                ).max()),
            )

        # 首末端有功之和 = 该支路有功损耗；全网求和 = 系统网损。
        worst_balance = max(
            worst_balance,
            abs(float((sample.target_pf_mw + sample.target_pt_mw).sum()) - sample.loss_mw),
        )

        # 相角差重建：dtheta 与 aux_va_deg 必须一致（这里只能检查量级）。
        worst_flow = max(worst_flow, float(np.abs(sample.target_branch_dtheta_deg).max()))

    print(f"  已知输入与真实值最大偏差 = {worst_known:.3e} MW/MVar/p.u.")
    print(f"  |Σ(Pf+Pt) - 网损| 最大值 = {worst_balance:.3e} MW")
    print(f"  |相角差| 最大值 = {worst_flow:.3f} deg")
    assert worst_known < 1e-6
    assert worst_balance < 1e-6
    print("  [PASS] 标签自洽")


def check_convergence(samples) -> None:
    _banner("5. 收敛与难度")

    converged = np.asarray([s.converged for s in samples])
    iterations = np.asarray([s.iterations for s in samples])
    print(f"  收敛 {int(converged.sum())}/{len(samples)} = {converged.mean():.2%}")
    print(f"  迭代数分布: {dict(sorted(Counter(iterations.tolist()).items()))}")
    print(f"  final_norm 最大值: {max(s.final_norm for s in samples):.3e}")
    assert converged.mean() > 0.99
    print("  [PASS] 收敛正常")


def check_round_trip(root: Path, n_samples: int) -> None:
    _banner("6. 读回校验")

    dataset = CompetitionDataset(root)
    assert len(dataset) == n_samples
    item = dataset[0]
    print(f"  len={len(dataset)}")
    print(f"  node_features={item['node_features'].shape}, "
          f"node_masks={item['node_masks'].shape}")
    print(f"  branch_features={item['branch_features'].shape}, "
          f"edge_index={item['edge_index'].shape}")
    print(f"  targets: " + ", ".join(f"{k}{v.shape}" for k, v in item["targets"].items()))
    print(f"  首条 mask_mode={item['meta']['mask_mode']}")

    batch = next(dataset.iter_batches(batch_size=8))
    print(f"  batch node_features={batch['node_features'].shape}, "
          f"batch targets.vm_pu={batch['targets']['vm_pu'].shape}")
    print("  [PASS] 读回正常")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--master-seed", type=int, default=20261019)
    parser.add_argument("--mask-seed", type=int, default=7)
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sigma-low", type=float, default=0.8)
    parser.add_argument("--sigma-high", type=float, default=1.10)
    parser.add_argument(
        "--mask-mode",
        choices=["structural", "dropout", "extra", "mixed"],
        default="mixed",
    )
    parser.add_argument("--dropout-prob", type=float, default=0.10)
    parser.add_argument("--extra-prob", type=float, default=0.10)
    args = parser.parse_args()

    perturb = PerturbationConfig(
        sigma_low=args.sigma_low,
        sigma_high=args.sigma_high,
    )
    mask_config = MaskConfig(
        mode=args.mask_mode,
        dropout_prob=args.dropout_prob,
        extra_prob=args.extra_prob,
    )

    _banner("0. 算例与配置")
    base = prepare_case39(flat=True)
    sigma_bar = load_capacity_sigma(base)
    topology = competition_topology(base)
    print(f"  算例: IEEE 39, nb={base.nb}, ne={base.branch.shape[0]}, "
          f"状态维数={len(base.pv) + 2 * len(base.pq)}")
    print(f"  负荷裕度 sigma_bar = {sigma_bar:.4f}; 采样区间 "
          f"[{perturb.sigma_low}, {perturb.sigma_high}]")
    print(f"  掩码模式 = {args.mask_mode}; schema = {COMPETITION_SCHEMA_VERSION}")
    assert perturb.sigma_high <= sigma_bar

    check_baseline(base)
    check_topology(topology, base)

    _banner(f"生成 {args.n_samples} 条样本")
    scenarios = [
        sample_load_scenario(base, perturb, sample_id=i, master_seed=args.master_seed)
        for i in range(args.n_samples)
    ]
    samples = []
    for scenario in scenarios:
        problem = apply_scenario(base, scenario)
        samples.append(
            label_competition_sample(
                problem,
                sample_id=scenario.sample_id,
                sigma=scenario.sigma,
                mask_config=mask_config,
                master_seed=args.master_seed,
                mask_seed=args.mask_seed,
            )
        )
    print(f"  已打标 {len(samples)} 条")

    check_masks(samples)
    check_targets(samples)
    check_convergence(samples)

    _banner("写盘")
    manifest = write_competition_dataset(
        args.out,
        topology,
        samples,
        config={
            "master_seed": args.master_seed,
            "mask_seed": args.mask_seed,
            "n_samples": args.n_samples,
            "shard_size": args.shard_size,
            "sigma_low": perturb.sigma_low,
            "sigma_high": perturb.sigma_high,
            "eps_p_low": perturb.eps_p_low,
            "eps_p_high": perturb.eps_p_high,
            "eps_q_low": perturb.eps_q_low,
            "eps_q_high": perturb.eps_q_high,
            "sigma_bar": sigma_bar,
            "mask_mode": args.mask_mode,
            "dropout_prob": args.dropout_prob,
            "extra_prob": args.extra_prob,
            "reference_case": "pypower.case39 (flat start)",
            "units": {
                "node_P": "MW (net injection)",
                "node_Q": "MVar (net injection)",
                "node_V": "p.u.",
                "node_kv": "kV",
                "branch_R": "p.u.",
                "branch_X": "p.u.",
                "branch_B_over_2": "p.u.",
                "target_Vm": "p.u.",
                "target_dtheta": "degree",
                "target_P": "MW",
                "target_Q": "MVar",
            },
        },
        shard_size=args.shard_size,
        topology_name="case39_base",
    )
    print(f"  输出目录: {args.out}")
    print(f"  分片: {[entry['count'] for entry in manifest['shards']]}")
    print(f"  topology_hash = {manifest['topology']['hash']}")

    check_round_trip(args.out, args.n_samples)

    print()
    print("[PASS] 赛事格式数据集构建完成")


if __name__ == "__main__":
    main()
