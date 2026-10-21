"""在测试集上重新评估一个 GNN checkpoint。

示例::

    python src/examples/evaluate_gnn.py --checkpoint runs/gnn_v1/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from powerflow.model import evaluate_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/gnn_v1/best.pt"))
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("dataset/ieee39/processed/competition_v1"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    torch.set_float32_matmul_precision("high")
    result = evaluate_checkpoint(
        args.checkpoint,
        args.data,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    payload = json.dumps(result, indent=2, ensure_ascii=False)
    print(payload)

    output = args.out or Path(args.checkpoint).with_name("metrics_recheck.json")
    output.write_text(payload + "\n", encoding="utf-8")
    print(f"\n已写入 {output}")


if __name__ == "__main__":
    main()
