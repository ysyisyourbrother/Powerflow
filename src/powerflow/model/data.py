"""把 ``CompetitionDataset`` 的样本整理成 GNN 可以消费的张量。

这一层只做四件事：

1. 只用训练集统计量拟合归一化器；
2. 把节点 P/Q/V 标准化，未知位置重新置 0；
3. 把支路 R/X/B2 标准化，支路类型转成 one-hot；
4. 按 ``case_hash`` 和场景难度做 80/10/10 划分，并打包成批。

数据集读取器本身保持原样，所有训练相关的处理都在这里完成。
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..dataset.competition import (
    MASK_KNOWN,
    MASK_PREDICT,
    CompetitionDataset,
)

# 与 ``CompetitionDataset`` 的字段顺序保持一致。
NODE_INPUT_DIM = 6  # P/Q/V + mask_P/mask_Q/mask_V
EDGE_INPUT_DIM = 6  # R/X/B2 + line/trafo/phase one-hot
NUM_BRANCH_TYPES = 3

# 输出通道顺序：这个顺序同时用于归一化、损失和评估，不能随意改。
TARGET_NAMES = (
    "vm_pu",
    "dtheta_deg",
    "pf_mw",
    "pt_mw",
    "qf_mvar",
    "qt_mvar",
)
BRANCH_TARGET_NAMES = TARGET_NAMES[1:]
TARGET_UNITS = {
    "vm_pu": "p.u.",
    "dtheta_deg": "deg",
    "pf_mw": "MW",
    "pt_mw": "MW",
    "qf_mvar": "MVar",
    "qt_mvar": "MVar",
}

DEFAULT_CLIP = 6.0


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


@dataclass
class Normalizer:
    """训练集统计量，负责输入标准化和输出反标准化。

    ``node_*`` 对应 P/Q/V，``branch_*`` 对应 R/X/B2，``target_*`` 对应
    六个输出通道。所有统计量都只在训练样本上计算。
    """

    node_mean: np.ndarray
    node_std: np.ndarray
    branch_mean: np.ndarray
    branch_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray
    clip: float = DEFAULT_CLIP

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        dataset: CompetitionDataset,
        indices: Sequence[int],
        *,
        clip: float = DEFAULT_CLIP,
    ) -> "Normalizer":
        if len(indices) == 0:
            raise ValueError("训练集为空，无法拟合归一化器")

        node_chunks: list[list[np.ndarray]] = [[], [], []]
        branch_chunks: list[np.ndarray] = []
        target_chunks: list[list[np.ndarray]] = [[] for _ in TARGET_NAMES]

        for raw_index in indices:
            item = dataset[int(raw_index)]
            values = _as_float_array(item["node_features"])
            masks = np.asarray(item["node_masks"], dtype=np.uint8)

            # 只统计已知位置。未知位置是 0 占位，不能参与均值/方差。
            for channel in range(3):
                known = masks[:, channel] == MASK_KNOWN
                if np.any(known):
                    node_chunks[channel].append(values[known, channel])

            branch_chunks.append(_as_float_array(item["branch_features"]))
            for channel, name in enumerate(TARGET_NAMES):
                target_chunks[channel].append(
                    _as_float_array(item["targets"][name]).reshape(-1)
                )

        node_mean = np.zeros(3, dtype=np.float64)
        node_std = np.ones(3, dtype=np.float64)
        for channel, chunks in enumerate(node_chunks):
            if chunks:
                samples = np.concatenate(chunks)
                node_mean[channel] = float(samples.mean())
                std = float(samples.std())
                node_std[channel] = std if std > 1e-9 else 1.0

        branch = np.concatenate(branch_chunks, axis=0)
        branch_mean = branch.mean(axis=0)
        branch_std = branch.std(axis=0)
        branch_std = np.where(branch_std > 1e-9, branch_std, 1.0)

        target_mean = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        target_std = np.ones(len(TARGET_NAMES), dtype=np.float64)
        for channel, chunks in enumerate(target_chunks):
            samples = np.concatenate(chunks)
            target_mean[channel] = float(samples.mean())
            std = float(samples.std())
            target_std[channel] = std if std > 1e-9 else 1.0

        return cls(
            node_mean=node_mean,
            node_std=node_std,
            branch_mean=branch_mean,
            branch_std=branch_std,
            target_mean=target_mean,
            target_std=target_std,
            clip=float(clip),
        )

    # ------------------------------------------------------------------
    # 输入变换
    # ------------------------------------------------------------------
    def transform_node(self, item: dict[str, Any]) -> np.ndarray:
        """返回 ``[N, 6]``：标准化 P/Q/V + 三个 0/1 掩码。"""

        values = _as_float_array(item["node_features"]).copy()
        masks = np.asarray(item["node_masks"], dtype=np.uint8)

        normalized = (values - self.node_mean) / self.node_std
        # 先标准化，再把未知位置清零。顺序不能反。
        normalized[masks == MASK_PREDICT] = 0.0
        np.clip(normalized, -self.clip, self.clip, out=normalized)

        return np.concatenate(
            [normalized, masks.astype(np.float32)], axis=-1
        ).astype(np.float32)

    def transform_branch(self, item: dict[str, Any]) -> np.ndarray:
        """返回 ``[E, 6]``：标准化 R/X/B2 + 三类 one-hot。"""

        features = _as_float_array(item["branch_features"])
        normalized = (features - self.branch_mean) / self.branch_std
        np.clip(normalized, -self.clip, self.clip, out=normalized)

        branch_type = np.asarray(item["branch_type"], dtype=np.int64)
        if np.any(branch_type < 0) or np.any(branch_type >= NUM_BRANCH_TYPES):
            raise ValueError(
                f"branch_type 必须在 0..{NUM_BRANCH_TYPES - 1} 之间，"
                f"实际为 {np.unique(branch_type).tolist()}"
            )
        one_hot = np.eye(NUM_BRANCH_TYPES, dtype=np.float32)[branch_type]
        return np.concatenate(
            [normalized.astype(np.float32), one_hot], axis=-1
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # 标签变换
    # ------------------------------------------------------------------
    def transform_targets(self, targets: dict[str, Any]) -> dict[str, np.ndarray]:
        transformed: dict[str, np.ndarray] = {}
        for channel, name in enumerate(TARGET_NAMES):
            values = _as_float_array(targets[name])
            transformed[name] = (
                (values - self.target_mean[channel]) / self.target_std[channel]
            ).astype(np.float32)
        return transformed

    def normalize_target(self, name: str, values: Any) -> np.ndarray:
        channel = TARGET_NAMES.index(name)
        return (
            _as_float_array(values) - self.target_mean[channel]
        ) / self.target_std[channel]

    def inverse_target(self, name: str, values: Any) -> np.ndarray:
        channel = TARGET_NAMES.index(name)
        return _as_float_array(values) * self.target_std[channel] + self.target_mean[channel]

    # ------------------------------------------------------------------
    # 保存与加载
    # ------------------------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "clip": self.clip,
            "target_names": list(TARGET_NAMES),
            "node_mean": self.node_mean.tolist(),
            "node_std": self.node_std.tolist(),
            "branch_mean": self.branch_mean.tolist(),
            "branch_std": self.branch_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Normalizer":
        names = tuple(state.get("target_names", ()))
        if names and names != TARGET_NAMES:
            raise ValueError(f"归一化器输出顺序不匹配：{names}")
        return cls(
            node_mean=np.asarray(state["node_mean"], dtype=np.float64),
            node_std=np.asarray(state["node_std"], dtype=np.float64),
            branch_mean=np.asarray(state["branch_mean"], dtype=np.float64),
            branch_std=np.asarray(state["branch_std"], dtype=np.float64),
            target_mean=np.asarray(state["target_mean"], dtype=np.float64),
            target_std=np.asarray(state["target_std"], dtype=np.float64),
            clip=float(state.get("clip", DEFAULT_CLIP)),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        return cls.from_state_dict(
            json.loads(Path(path).read_text(encoding="utf-8"))
        )


# ----------------------------------------------------------------------
# 单样本与批打包
# ----------------------------------------------------------------------
def build_graph_item(
    item: dict[str, Any], normalizer: Normalizer
) -> dict[str, Any]:
    """把一个原始样本转换成训练用的图张量字典。

    坐标约定：

    * ``node_x`` ``[N, 6]``
    * ``edge_index`` ``[E, 2]``，每行是 ``(from, to)``
    * ``edge_attr`` ``[E, 6]``
    """

    node_values = _as_float_array(item["node_features"])
    masks = np.asarray(item["node_masks"], dtype=np.uint8)

    node_x = normalizer.transform_node(item)
    edge_attr = normalizer.transform_branch(item)
    edge_index = np.ascontiguousarray(
        np.asarray(item["edge_index"], dtype=np.int64).T
    )

    known_v_mask = masks[:, 2] == MASK_KNOWN
    known_v_norm = normalizer.normalize_target("vm_pu", node_values[:, 2])
    known_v_norm = np.where(known_v_mask, known_v_norm, 0.0).astype(np.float32)

    num_nodes = node_x.shape[0]
    num_edges = edge_attr.shape[0]

    return {
        "node_x": node_x,
        "node_valid": np.ones(num_nodes, dtype=bool),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_valid": np.ones(num_edges, dtype=bool),
        "branch_type": np.asarray(item["branch_type"], dtype=np.int64),
        "known_v_norm": known_v_norm,
        "known_v_mask": known_v_mask,
        "v_predict_mask": masks[:, 2] == MASK_PREDICT,
        "targets": normalizer.transform_targets(item["targets"]),
        "meta": dict(item["meta"]),
    }


def collate_graphs(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """把不同大小的图补零成 ``[B, N, ...]`` 和 ``[B, E, ...]``。

    当前 IEEE 39 数据所有样本都是 39 节点、46 支路，补零只是为以后
    多拓扑数据留出接口。
    """

    if not items:
        raise ValueError("不能对空列表做 collate")

    batch_size = len(items)
    max_nodes = max(item["node_x"].shape[0] for item in items)
    max_edges = max(item["edge_attr"].shape[0] for item in items)

    node_x = np.zeros((batch_size, max_nodes, NODE_INPUT_DIM), dtype=np.float32)
    node_valid = np.zeros((batch_size, max_nodes), dtype=bool)
    edge_index = np.zeros((batch_size, max_edges, 2), dtype=np.int64)
    edge_attr = np.zeros((batch_size, max_edges, EDGE_INPUT_DIM), dtype=np.float32)
    edge_valid = np.zeros((batch_size, max_edges), dtype=bool)
    branch_type = np.full((batch_size, max_edges), -1, dtype=np.int64)

    known_v_norm = np.zeros((batch_size, max_nodes), dtype=np.float32)
    known_v_mask = np.zeros((batch_size, max_nodes), dtype=bool)
    v_predict_mask = np.zeros((batch_size, max_nodes), dtype=bool)

    targets = {
        name: np.zeros(
            (batch_size, max_nodes if name == "vm_pu" else max_edges),
            dtype=np.float32,
        )
        for name in TARGET_NAMES
    }

    meta_keys = tuple(items[0]["meta"].keys())

    for batch_index, item in enumerate(items):
        num_nodes = item["node_x"].shape[0]
        num_edges = item["edge_attr"].shape[0]

        node_x[batch_index, :num_nodes] = item["node_x"]
        node_valid[batch_index, :num_nodes] = item["node_valid"]
        edge_index[batch_index, :num_edges] = item["edge_index"]
        edge_attr[batch_index, :num_edges] = item["edge_attr"]
        edge_valid[batch_index, :num_edges] = item["edge_valid"]
        branch_type[batch_index, :num_edges] = item["branch_type"]

        known_v_norm[batch_index, :num_nodes] = item["known_v_norm"]
        known_v_mask[batch_index, :num_nodes] = item["known_v_mask"]
        v_predict_mask[batch_index, :num_nodes] = item["v_predict_mask"]

        for name in TARGET_NAMES:
            size = num_nodes if name == "vm_pu" else num_edges
            targets[name][batch_index, :size] = item["targets"][name]

    batch = {
        "node_x": node_x,
        "node_valid": node_valid,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_valid": edge_valid,
        "branch_type": branch_type,
        "known_v_norm": known_v_norm,
        "known_v_mask": known_v_mask,
        "v_predict_mask": v_predict_mask,
        "targets": targets,
        "meta": {
            key: np.asarray([item["meta"][key] for item in items])
            for key in meta_keys
        },
    }
    return batch


def _to_torch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _to_torch(item, device) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "USO":
            return value
        return torch.as_tensor(value, device=device)
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return torch.as_tensor(value, device=device)
    return value


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把数值数组转成张量并搬到设备；``meta`` 保留为 numpy 供分析使用。"""

    converted: dict[str, Any] = {}
    for key, value in batch.items():
        if key == "meta":
            converted[key] = value
        else:
            converted[key] = _to_torch(value, device)
    return converted


class CompetitionGraphDataset(Dataset):
    """在 ``CompetitionDataset`` 外面套一层归一化和图张量构造。"""

    def __init__(
        self,
        dataset: CompetitionDataset,
        indices: Sequence[int],
        normalizer: Normalizer,
    ) -> None:
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.normalizer = normalizer

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return build_graph_item(
            self.dataset[int(self.indices[index])], self.normalizer
        )


def make_dataloader(
    dataset: CompetitionDataset,
    indices: Sequence[int],
    normalizer: Normalizer,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    graph_dataset = CompetitionGraphDataset(dataset, indices, normalizer)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        graph_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_graphs,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator if shuffle else None,
    )


# ----------------------------------------------------------------------
# 数据划分
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    """按 ``case_hash`` 分组、按掩码模式和负荷水平分层的划分配置。"""

    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    seed: int = 20261021
    sigma_bins: int = 5

    def __post_init__(self) -> None:
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"划分比例之和必须为 1，当前为 {total}")
        if self.sigma_bins < 1:
            raise ValueError("sigma_bins 必须为正")


def _sigma_bins(sigma: np.ndarray, num_bins: int) -> np.ndarray:
    """按分位数把负荷水平分成近似等量的桶。"""

    if num_bins <= 1:
        return np.zeros_like(sigma, dtype=np.int64)
    edges = np.quantile(sigma, np.linspace(0.0, 1.0, num_bins + 1))
    edges = np.unique(edges)
    if edges.size <= 2:
        return np.zeros_like(sigma, dtype=np.int64)
    inner = edges[1:-1]
    return np.searchsorted(inner, sigma, side="right").astype(np.int64)


def make_split(
    dataset: CompetitionDataset, config: SplitConfig | None = None
) -> dict[str, Any]:
    """返回 ``{"train": [...], "val": [...], "test": [...]}``。

    同一个 ``case_hash`` 的样本只会进入一个集合。每个 ``(mask_mode, sigma
    分位)`` 分层内部再按组随机划分，尽量让三个集合都覆盖三种掩码和
    不同负荷水平。
    """

    config = config or SplitConfig()
    metadata = dataset.metadata()

    case_hashes = np.asarray(metadata["case_hash"])
    mask_modes = np.asarray(metadata["mask_mode"], dtype=np.int64)
    sigmas = np.asarray(metadata["sigma"], dtype=np.float64)
    sigma_bins = _sigma_bins(sigmas, config.sigma_bins)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, case_hash in enumerate(case_hashes.tolist()):
        groups[str(case_hash)].append(index)

    strata: dict[tuple[int, int], list[list[int]]] = defaultdict(list)
    for members in groups.values():
        keys = Counter(
            (int(mask_modes[index]), int(sigma_bins[index])) for index in members
        )
        stratum = keys.most_common(1)[0][0]
        strata[stratum].append(members)

    split_indices: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for stratum, group_list in sorted(strata.items()):
        # 每个分层用自己的随机流，整体划分可复现。
        rng = np.random.default_rng(config.seed + stratum[0] * 100 + stratum[1])
        order = rng.permutation(len(group_list)).tolist()
        shuffled = [group_list[index] for index in order]

        num_groups = len(shuffled)
        if num_groups >= 3:
            num_test = max(1, int(round(num_groups * config.test_fraction)))
            num_val = max(1, int(round(num_groups * config.val_fraction)))
            if num_test + num_val >= num_groups:
                num_test, num_val = 1, 1
            num_train = num_groups - num_test - num_val
        elif num_groups == 2:
            num_train, num_val, num_test = 1, 1, 0
        else:
            num_train, num_val, num_test = 1, 0, 0

        partitions = {
            "train": shuffled[:num_train],
            "val": shuffled[num_train : num_train + num_val],
            "test": shuffled[num_train + num_val : num_train + num_val + num_test],
        }
        for name, group_chunk in partitions.items():
            for members in group_chunk:
                split_indices[name].extend(members)

    for name in split_indices:
        split_indices[name] = sorted(split_indices[name])

    all_indices = set(range(len(dataset)))
    train_set = set(split_indices["train"])
    val_set = set(split_indices["val"])
    test_set = set(split_indices["test"])
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("数据划分出现重叠")
    if train_set | val_set | test_set != all_indices:
        raise RuntimeError("数据划分没有覆盖全部样本")

    return {
        "version": 1,
        "config": {
            "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
            "test_fraction": config.test_fraction,
            "seed": config.seed,
            "sigma_bins": config.sigma_bins,
        },
        "train": split_indices["train"],
        "val": split_indices["val"],
        "test": split_indices["test"],
        "counts": {name: len(values) for name, values in split_indices.items()},
        "sample_ids": {
            name: np.asarray(metadata["sample_id"])[values].tolist()
            for name, values in split_indices.items()
        },
    }


def save_split(split: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(split, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def load_split(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
