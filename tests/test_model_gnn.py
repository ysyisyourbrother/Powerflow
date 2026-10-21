"""GNN 数据管线、前向形状、已知电压复制和训练冒烟测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from powerflow.model import (  # noqa: E402
    Normalizer,
    PowerFlowGNN,
    PowerFlowGNNConfig,
    SplitConfig,
    batch_to_device,
    build_graph_item,
    collate_graphs,
    compute_loss,
    make_split,
    train_model,
)


def make_item(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    num_nodes = 5
    num_edges = 4

    p = rng.normal(0.0, 120.0, size=num_nodes)
    q = rng.normal(0.0, 40.0, size=num_nodes)
    v = 1.0 + rng.normal(0.0, 0.02, size=num_nodes)

    masks = np.zeros((num_nodes, 3), dtype=np.uint8)
    masks[0, 0] = 1
    masks[1, 1] = 1
    masks[2, 2] = 1

    values = np.stack([p, q, v], axis=-1)
    values[masks == 1] = 0.0

    edge_index = np.asarray(
        [[0, 1, 2, 3], [1, 2, 3, 4]],
        dtype=np.int64,
    )
    branch_features = np.asarray(
        [
            [0.001, 0.02, 0.01],
            [0.002, 0.03, 0.00],
            [0.000, 0.01, 0.00],
            [0.000, 0.02, 0.00],
        ],
        dtype=np.float64,
    )
    branch_type = np.asarray([0, 0, 1, 1], dtype=np.int64)

    target_vm = v.copy()
    target_vm[masks[:, 2] == 1] = 1.0 + rng.normal(0.0, 0.01, size=(masks[:, 2] == 1).sum())

    return {
        "node_features": values,
        "node_masks": masks,
        "edge_index": edge_index,
        "branch_features": branch_features,
        "branch_type": branch_type,
        "targets": {
            "vm_pu": target_vm,
            "dtheta_deg": rng.normal(0.0, 2.0, size=num_edges),
            "pf_mw": rng.normal(0.0, 80.0, size=num_edges),
            "pt_mw": rng.normal(0.0, 80.0, size=num_edges),
            "qf_mvar": rng.normal(0.0, 20.0, size=num_edges),
            "qt_mvar": rng.normal(0.0, 20.0, size=num_edges),
        },
        "meta": {
            "sample_id": np.uint64(seed),
            "mask_mode": np.int8(seed % 3),
            "sigma": np.float64(0.9 + 0.01 * seed),
            "case_hash": f"case-{seed:04d}",
        },
    }


class FakeDataset:
    def __init__(self, count: int = 12) -> None:
        self.items = [make_item(seed) for seed in range(count)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        return self.items[index]

    def metadata(self) -> dict[str, np.ndarray]:
        return {
            "sample_id": np.asarray(
                [item["meta"]["sample_id"] for item in self.items], dtype=np.uint64
            ),
            "mask_mode": np.asarray(
                [item["meta"]["mask_mode"] for item in self.items], dtype=np.int8
            ),
            "sigma": np.asarray(
                [item["meta"]["sigma"] for item in self.items], dtype=np.float64
            ),
            "case_hash": np.asarray(
                [item["meta"]["case_hash"] for item in self.items]
            ),
        }


def make_batch(count: int = 3) -> tuple[dict, Normalizer, list[dict]]:
    dataset = FakeDataset(count)
    normalizer = Normalizer.fit(dataset, list(range(count)))
    items = [build_graph_item(dataset[index], normalizer) for index in range(count)]
    return collate_graphs(items), normalizer, items


def test_normalizer_zeroes_unknown_and_keeps_masks():
    dataset = FakeDataset(1)
    normalizer = Normalizer.fit(dataset, [0])
    item = build_graph_item(dataset[0], normalizer)

    # 未知位置必须重新变成 0，掩码仍然保留。
    unknown = dataset[0]["node_masks"] == 1
    assert np.all(item["node_x"][..., :3][unknown] == 0.0)
    assert np.all(item["node_x"][..., 3:] == dataset[0]["node_masks"])


def test_graph_batch_shapes():
    batch, _, _ = make_batch(3)
    assert batch["node_x"].shape == (3, 5, 6)
    assert batch["node_valid"].shape == (3, 5)
    assert batch["edge_index"].shape == (3, 4, 2)
    assert batch["edge_attr"].shape == (3, 4, 6)
    assert batch["edge_valid"].shape == (3, 4)
    assert batch["targets"]["vm_pu"].shape == (3, 5)
    assert batch["targets"]["pf_mw"].shape == (3, 4)


def test_model_forward_shapes_and_known_voltage_copy():
    batch, normalizer, _ = make_batch(3)
    batch = batch_to_device(batch, torch.device("cpu"))
    config = PowerFlowGNNConfig(hidden=16, num_layers=2, dropout=0.0)
    model = PowerFlowGNN(config)
    model.set_target_stats(normalizer.target_mean, normalizer.target_std)

    normalized = model(batch)
    assert normalized["vm_pu"].shape == (3, 5)
    assert normalized["dtheta_deg"].shape == (3, 4)
    assert normalized["pf_mw"].shape == (3, 4)
    assert normalized["pt_mw"].shape == (3, 4)
    assert normalized["qf_mvar"].shape == (3, 4)
    assert normalized["qt_mvar"].shape == (3, 4)

    physical = model.forward_physical(batch)
    assert physical["vm_pu"].shape == (3, 5)

    # 已知电压在输出阶段直接复制输入；用物理量比较。
    known = batch["known_v_mask"].numpy()
    known_values = (
        batch["known_v_norm"].numpy() * normalizer.target_std[0]
        + normalizer.target_mean[0]
    )
    predicted = physical["vm_pu"].detach().numpy()
    assert np.allclose(predicted[known], known_values[known], atol=1e-5)


def test_loss_is_finite_and_backward_produces_gradients():
    batch, normalizer, _ = make_batch(3)
    batch = batch_to_device(batch, torch.device("cpu"))
    model = PowerFlowGNN(PowerFlowGNNConfig(hidden=16, num_layers=2, dropout=0.0))
    model.set_target_stats(normalizer.target_mean, normalizer.target_std)

    loss, parts = compute_loss(model, batch)
    assert torch.isfinite(loss)
    assert set(parts) == {
        "vm_pu",
        "dtheta_deg",
        "pf_mw",
        "pt_mw",
        "qf_mvar",
        "qt_mvar",
    }
    loss.backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in model.parameters()
    )


def test_split_groups_case_hash_and_covers_all_samples():
    count = 90
    dataset = FakeDataset(count)
    # 每两个样本共享一个 case_hash，验证同组不会被拆开。
    for index, item in enumerate(dataset.items):
        item["meta"]["case_hash"] = f"case-{index // 2:04d}"
        item["meta"]["mask_mode"] = np.int8((index // 2) % 3)
        item["meta"]["sigma"] = np.float64(0.8 + 0.005 * (index // 2))

    split = make_split(dataset, SplitConfig(seed=123))
    train = set(split["train"])
    val = set(split["val"])
    test = set(split["test"])

    assert not (train & val)
    assert not (train & test)
    assert not (val & test)
    assert train | val | test == set(range(count))
    assert all(len(split[name]) > 0 for name in ("train", "val", "test"))

    hashes_by_split = {}
    for name in ("train", "val", "test"):
        hashes_by_split[name] = {
            dataset.items[index]["meta"]["case_hash"] for index in split[name]
        }
    assert not (hashes_by_split["train"] & hashes_by_split["val"])
    assert not (hashes_by_split["train"] & hashes_by_split["test"])
    assert not (hashes_by_split["val"] & hashes_by_split["test"])


def test_train_model_smoke(tmp_path: Path):
    dataset = FakeDataset(12)
    normalizer = Normalizer.fit(dataset, list(range(8)))
    model = PowerFlowGNN(PowerFlowGNNConfig(hidden=16, num_layers=2, dropout=0.0))
    model.set_target_stats(normalizer.target_mean, normalizer.target_std)

    split = {
        "train": list(range(8)),
        "val": [8, 9],
        "test": [10, 11],
        "counts": {"train": 8, "val": 2, "test": 2},
    }
    result = train_model(
        dataset,
        split,
        normalizer,
        model,
        torch.device("cpu"),
        tmp_path,
        epochs=2,
        batch_size=4,
        patience=5,
        num_workers=0,
        log_every=10,
    )

    assert result["best_epoch"] >= 1
    assert (tmp_path / "best.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "normalizer.json").exists()
    assert result["metrics"]["vm_pu"]["all"]["count"] == 2 * 5
