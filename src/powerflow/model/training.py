"""训练、评估和指标统计。"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..dataset.competition import CompetitionDataset, MASK_MODE_NAMES
from .data import (
    BRANCH_TARGET_NAMES,
    TARGET_NAMES,
    TARGET_UNITS,
    Normalizer,
    batch_to_device,
    make_dataloader,
)
from .gnn import PowerFlowGNN, PowerFlowGNNConfig


DEFAULT_LOSS_WEIGHTS: dict[str, float] = {name: 1.0 for name in TARGET_NAMES}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    model: PowerFlowGNN,
    batch: dict[str, Any],
    loss_weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """六通道 Huber 损失。

    Vm 只在“待预测”节点上计损失；已知电压在模型输出阶段已经直接复制。
    五个支路量在所有有效支路上计损失。所有量都在标准化空间里比较，
    这样 MW 级功率不会压过 p.u. 级电压和度级相角。
    """

    weights = loss_weights or DEFAULT_LOSS_WEIGHTS
    predictions = model(batch)
    targets = batch["targets"]
    node_valid = batch["node_valid"]
    edge_valid = batch["edge_valid"]

    vm_mask = batch["v_predict_mask"] & node_valid
    zero = predictions["vm_pu"].sum() * 0.0

    if bool(vm_mask.any()):
        loss_vm = F.smooth_l1_loss(
            predictions["vm_pu"][vm_mask], targets["vm_pu"][vm_mask], beta=1.0
        )
    else:
        loss_vm = zero

    parts: dict[str, torch.Tensor] = {"vm_pu": loss_vm}
    loss_branch = zero
    for name in BRANCH_TARGET_NAMES:
        if bool(edge_valid.any()):
            value = F.smooth_l1_loss(
                predictions[name][edge_valid], targets[name][edge_valid], beta=1.0
            )
        else:
            value = zero
        parts[name] = value
        loss_branch = loss_branch + weights[name] * value

    total = weights["vm_pu"] * loss_vm + loss_branch
    return total, {name: float(value.detach()) for name, value in parts.items()}


def train_one_epoch(
    model: PowerFlowGNN,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    loss_weights: dict[str, float] | None = None,
    grad_clip: float = 1.0,
    amp: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    amp_enabled = amp and device.type == "cuda"

    for batch in loader:
        batch = batch_to_device(batch, device)
        batch_size = batch["node_x"].shape[0]

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss, _ = compute_loss(model, batch, loss_weights)

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate_loss(
    model: PowerFlowGNN,
    loader: Any,
    device: torch.device,
    *,
    loss_weights: dict[str, float] | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        batch_size = batch["node_x"].shape[0]
        loss, _ = compute_loss(model, batch, loss_weights)
        total_loss += float(loss) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1)


def _as_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def collect_predictions(
    model: PowerFlowGNN,
    loader: Any,
    device: torch.device,
    normalizer: Normalizer,
) -> dict[str, Any]:
    """把所有批次的物理量预测拼成扁平数组，供指标统计和误差分析使用。"""

    model.eval()

    node_pred: list[np.ndarray] = []
    node_target: list[np.ndarray] = []
    node_vpredict: list[np.ndarray] = []
    node_mask_mode: list[np.ndarray] = []
    node_sigma: list[np.ndarray] = []

    branch_pred: dict[str, list[np.ndarray]] = {
        name: [] for name in BRANCH_TARGET_NAMES
    }
    branch_target: dict[str, list[np.ndarray]] = {
        name: [] for name in BRANCH_TARGET_NAMES
    }
    branch_type: list[np.ndarray] = []
    branch_mask_mode: list[np.ndarray] = []
    branch_sigma: list[np.ndarray] = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        prediction = model.forward_physical(batch)
        normalized_targets = batch["targets"]

        node_valid = _as_numpy(batch["node_valid"]).astype(bool)
        vm_pred = _as_numpy(prediction["vm_pu"])
        vm_target = normalizer.inverse_target(
            "vm_pu", _as_numpy(normalized_targets["vm_pu"])
        )
        v_predict = _as_numpy(batch["v_predict_mask"]).astype(bool)

        node_pred.append(vm_pred[node_valid])
        node_target.append(vm_target[node_valid])
        node_vpredict.append(v_predict[node_valid])

        node_counts = node_valid.sum(axis=1)
        node_mask_mode.append(
            np.repeat(_as_numpy(batch["meta"]["mask_mode"]), node_counts)
        )
        node_sigma.append(
            np.repeat(_as_numpy(batch["meta"]["sigma"]).astype(np.float64), node_counts)
        )

        edge_valid = _as_numpy(batch["edge_valid"]).astype(bool)
        edge_counts = edge_valid.sum(axis=1)
        for name in BRANCH_TARGET_NAMES:
            branch_pred[name].append(_as_numpy(prediction[name])[edge_valid])
            branch_target[name].append(
                normalizer.inverse_target(
                    name, _as_numpy(normalized_targets[name])
                )[edge_valid]
            )

        branch_type.append(_as_numpy(batch["branch_type"])[edge_valid])
        branch_mask_mode.append(
            np.repeat(_as_numpy(batch["meta"]["mask_mode"]), edge_counts)
        )
        branch_sigma.append(
            np.repeat(_as_numpy(batch["meta"]["sigma"]).astype(np.float64), edge_counts)
        )

    def concatenate(chunks: list[np.ndarray], dtype: Any = None) -> np.ndarray:
        if not chunks:
            return np.asarray([], dtype=dtype)
        return np.concatenate(chunks).astype(dtype) if dtype is not None else np.concatenate(chunks)

    return {
        "node": {
            "pred": concatenate(node_pred),
            "target": concatenate(node_target),
            "v_predict": concatenate(node_vpredict, dtype=bool),
            "mask_mode": concatenate(node_mask_mode, dtype=np.int64),
            "sigma": concatenate(node_sigma, dtype=np.float64),
        },
        "branch": {
            "pred": {
                name: concatenate(branch_pred[name]) for name in BRANCH_TARGET_NAMES
            },
            "target": {
                name: concatenate(branch_target[name]) for name in BRANCH_TARGET_NAMES
            },
            "branch_type": concatenate(branch_type, dtype=np.int64),
            "mask_mode": concatenate(branch_mask_mode, dtype=np.int64),
            "sigma": concatenate(branch_sigma, dtype=np.float64),
        },
    }


def _metric(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    if count == 0:
        return {"count": 0, "mae": None, "rmse": None, "max_abs": None}
    error = pred[mask] - target[mask]
    return {
        "count": count,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs": float(np.max(np.abs(error))),
    }


def _high_load_mask(sigma: np.ndarray) -> np.ndarray:
    if sigma.size == 0:
        return np.zeros_like(sigma, dtype=bool)
    threshold = float(np.quantile(sigma, 0.8))
    return sigma >= threshold


def compute_metrics(collected: dict[str, Any]) -> dict[str, Any]:
    """按总体、掩码模式、负荷水平、支路类型拆分评估误差。"""

    node = collected["node"]
    branch = collected["branch"]

    node_all = np.ones_like(node["pred"], dtype=bool)
    high_load_nodes = _high_load_mask(node["sigma"])

    node_metrics: dict[str, Any] = {
        "all": _metric(node["pred"], node["target"], node_all),
        "unknown": _metric(node["pred"], node["target"], node["v_predict"]),
        "known": _metric(node["pred"], node["target"], ~node["v_predict"]),
        "high_load": _metric(node["pred"], node["target"], high_load_nodes),
        "by_mask_mode": {},
    }
    for mode_value, mode_name in MASK_MODE_NAMES.items():
        mask = node["mask_mode"] == mode_value
        node_metrics["by_mask_mode"][mode_name] = _metric(
            node["pred"], node["target"], mask
        )

    high_load_branches = _high_load_mask(branch["sigma"])
    branch_metrics: dict[str, Any] = {}
    for name in BRANCH_TARGET_NAMES:
        pred = branch["pred"][name]
        target = branch["target"][name]
        all_mask = np.ones_like(pred, dtype=bool)
        metrics = {
            "all": _metric(pred, target, all_mask),
            "line": _metric(pred, target, branch["branch_type"] == 0),
            "transformer": _metric(pred, target, branch["branch_type"] == 1),
            "phase_shifter": _metric(pred, target, branch["branch_type"] == 2),
            "high_load": _metric(pred, target, high_load_branches),
            "by_mask_mode": {},
        }
        for mode_value, mode_name in MASK_MODE_NAMES.items():
            metrics["by_mask_mode"][mode_name] = _metric(
                pred, target, branch["mask_mode"] == mode_value
            )
        branch_metrics[name] = metrics

    return {
        "vm_pu": {
            **node_metrics,
            "unit": TARGET_UNITS["vm_pu"],
        },
        **{
            name: {
                **branch_metrics[name],
                "unit": TARGET_UNITS[name],
            }
            for name in BRANCH_TARGET_NAMES
        },
    }


def _write_json(path: Path, payload: Any) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"无法序列化 {type(value)!r}")

    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=default),
        encoding="utf-8",
    )


def save_checkpoint(
    path: Path,
    model: PowerFlowGNN,
    normalizer: Normalizer,
    split: dict[str, Any],
    *,
    epoch: int,
    val_loss: float,
    args: dict[str, Any] | None = None,
) -> None:
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "normalizer": normalizer.state_dict(),
            "split": split,
            "target_names": list(TARGET_NAMES),
            "epoch": int(epoch),
            "val_loss": float(val_loss),
            "args": args or {},
        },
        path,
    )


def train_model(
    dataset: CompetitionDataset,
    split: dict[str, Any],
    normalizer: Normalizer,
    model: PowerFlowGNN,
    device: torch.device,
    out_dir: str | Path,
    *,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 30,
    num_workers: int = 0,
    amp: bool = False,
    grad_clip: float = 1.0,
    loss_weights: dict[str, float] | None = None,
    seed: int = 20261021,
    log_every: int = 1,
) -> dict[str, Any]:
    """完整训练循环：训练、验证、早停、测试评估和产物落盘。"""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not split.get("train"):
        raise ValueError("训练集为空")
    if not split.get("val"):
        raise ValueError("验证集为空，无法做早停")

    normalizer.save(output / "normalizer.json")
    _write_json(output / "split.json", split)
    _write_json(
        output / "config.json",
        {
            "model": asdict(model.config),
            "training": {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "weight_decay": weight_decay,
                "patience": patience,
                "amp": amp,
                "grad_clip": grad_clip,
                "seed": seed,
                "loss_weights": loss_weights or DEFAULT_LOSS_WEIGHTS,
            },
        },
    )

    train_loader = make_dataloader(
        dataset,
        split["train"],
        normalizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        seed=seed,
    )
    val_loader = make_dataloader(
        dataset,
        split["val"],
        normalizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = make_dataloader(
        dataset,
        split["test"],
        normalizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, patience // 3), min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")

    best_val = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        started = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights=loss_weights,
            grad_clip=grad_clip,
            amp=amp,
            scaler=scaler,
        )
        val_loss = evaluate_loss(
            model, val_loader, device, loss_weights=loss_weights
        )
        scheduler.step(val_loss)
        elapsed = time.time() - started
        current_lr = float(optimizer.param_groups[0]["lr"])

        improved = val_loss < best_val - 1e-9
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            save_checkpoint(
                output / "best.pt",
                model,
                normalizer,
                split,
                epoch=epoch,
                val_loss=val_loss,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "best_val": best_val,
                "seconds": elapsed,
            }
        )
        if epoch == 1 or epoch % max(log_every, 1) == 0 or improved:
            marker = " *" if improved else ""
            print(
                f"epoch {epoch:4d}/{epochs}  train {train_loss:.5f}  "
                f"val {val_loss:.5f}  lr {current_lr:.2e}{marker}"
            )

        if bad_epochs >= patience:
            print(f"early stop at epoch {epoch}（最佳 epoch {best_epoch}）")
            break

    save_checkpoint(
        output / "last.pt",
        model,
        normalizer,
        split,
        epoch=history[-1]["epoch"] if history else 0,
        val_loss=history[-1]["val_loss"] if history else float("inf"),
    )

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    collected = collect_predictions(model, test_loader, device, normalizer)
    metrics = compute_metrics(collected)
    _write_json(output / "history.json", history)
    _write_json(
        output / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "test": metrics,
        },
    )

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "history": history,
        "metrics": metrics,
        "output_dir": str(output),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | Path,
    device: torch.device,
    *,
    batch_size: int = 128,
    num_workers: int = 0,
) -> dict[str, Any]:
    """加载一个 checkpoint，在它保存的测试集上重新评估。"""

    checkpoint = torch.load(
        Path(checkpoint_path), map_location="cpu", weights_only=False
    )
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    model = PowerFlowGNN(PowerFlowGNNConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.set_target_stats(normalizer.target_mean, normalizer.target_std)
    model.to(device)

    dataset = CompetitionDataset(data_root, cache_shards=True)
    split = checkpoint["split"]
    test_loader = make_dataloader(
        dataset,
        split["test"],
        normalizer,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    collected = collect_predictions(model, test_loader, device, normalizer)
    return {
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("val_loss"),
        "test": compute_metrics(collected),
    }
