"""训练赛事对齐的 GNN 潮流预测模型。

示例::

    # 快速冒烟：每边只取少量样本
    python src/examples/train_gnn.py --max-train-samples 64 --max-val-samples 32 \
        --max-test-samples 32 --epochs 5

    # 正常训练（RTX 5090 可直接用 cuda）
    python src/examples/train_gnn.py --device cuda --epochs 200

产物默认写到 ``runs/gnn_v1/``：

* ``best.pt`` / ``last.pt``：模型、归一化器、划分和反标准化参数；
* ``normalizer.json``：训练集统计量；
* ``split.json``：训练/验证/测试下标；
* ``history.json``：逐轮损失；
* ``metrics.json``：测试集分项指标。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow.dataset import CompetitionDataset  # noqa: E402
from powerflow.model import (  # noqa: E402
    Normalizer,
    PowerFlowGNN,
    PowerFlowGNNConfig,
    SplitConfig,
    load_split,
    make_split,
    save_split,
    set_seed,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/ieee39/processed/competition_v1"),
        help="competition_v1 数据集目录",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs/gnn_v1"),
        help="训练产物输出目录",
    )
    parser.add_argument("--split", type=Path, default=None, help="复用已有的 split.json")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0")
    parser.add_argument("--amp", action="store_true", help="使用 CUDA 混合精度")
    parser.add_argument(
        "--no-copy-known-v", action="store_true", help="关闭已知电压直接复制"
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--max-train-samples", type=int, default=None, help="仅用于冒烟测试"
    )
    parser.add_argument(
        "--max-val-samples", type=int, default=None, help="仅用于冒烟测试"
    )
    parser.add_argument(
        "--max-test-samples", type=int, default=None, help="仅用于冒烟测试"
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def truncate_split(split: dict, args: argparse.Namespace) -> dict:
    limits = {
        "train": args.max_train_samples,
        "val": args.max_val_samples,
        "test": args.max_test_samples,
    }
    if not any(value is not None for value in limits.values()):
        return split

    trimmed = dict(split)
    for name, limit in limits.items():
        if limit is not None:
            trimmed[name] = list(split[name])[: int(limit)]
    trimmed["counts"] = {name: len(trimmed[name]) for name in ("train", "val", "test")}
    if "sample_ids" in split:
        trimmed["sample_ids"] = {
            name: list(split["sample_ids"][name])[: len(trimmed[name])]
            for name in ("train", "val", "test")
        }
    return trimmed


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")

    dataset = CompetitionDataset(args.data, cache_shards=True)
    print(f"数据集: {args.data}  样本数: {len(dataset)}")

    if args.split is not None:
        split = load_split(args.split)
        print(f"复用划分: {args.split}")
    else:
        split = make_split(dataset, SplitConfig(seed=args.seed))
        print(
            "新建划分: "
            + ", ".join(f"{name}={len(split[name])}" for name in ("train", "val", "test"))
        )
    split = truncate_split(split, args)
    args.out.mkdir(parents=True, exist_ok=True)
    save_split(split, args.out / "split.json")

    normalizer = Normalizer.fit(dataset, split["train"])
    print(
        "归一化: "
        f"node mean={normalizer.node_mean.round(3).tolist()} "
        f"target mean={normalizer.target_mean.round(3).tolist()}"
    )

    config = PowerFlowGNNConfig(
        hidden=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        copy_known_v=not args.no_copy_known_v,
    )
    model = PowerFlowGNN(config)
    model.set_target_stats(normalizer.target_mean, normalizer.target_std)
    model.to(device)

    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"设备: {device}  模型参数: {num_parameters:,}")

    result = train_model(
        dataset,
        split,
        normalizer,
        model,
        device,
        args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        num_workers=args.num_workers,
        amp=args.amp,
        grad_clip=args.grad_clip,
        seed=args.seed,
        log_every=args.log_every,
    )

    metrics = result["metrics"]
    print()
    print("=" * 72)
    print(f"最佳 epoch: {result['best_epoch']}  验证损失: {result['best_val_loss']:.6f}")
    for name in ("vm_pu", "dtheta_deg", "pf_mw", "pt_mw", "qf_mvar", "qt_mvar"):
        item = metrics[name]
        subset = item["unknown"] if name == "vm_pu" else item["all"]
        unit = item["unit"]
        print(
            f"{name:12s} {subset['mae']:9.4f} {unit:5s} "
            f"RMSE {subset['rmse']:9.4f}  n={subset['count']}"
        )
    print(f"产物目录: {result['output_dir']}")


if __name__ == "__main__":
    main()
