"""把潮流结果画成网络结构和母线电压图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import networkx as nx
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from pypower.idx_brch import BR_STATUS, F_BUS, PF, PT, RATE_A, T_BUS
from pypower.idx_bus import BUS_I, BUS_TYPE, PQ, PV, REF, VM


def _layout(branch: np.ndarray, bus_numbers: np.ndarray) -> dict[int, np.ndarray]:
    graph = nx.Graph()
    graph.add_nodes_from(bus_numbers.tolist())
    for row in branch:
        if row[BR_STATUS] > 0:
            graph.add_edge(int(row[F_BUS]), int(row[T_BUS]))
    return nx.spring_layout(graph, seed=7, k=1.2, iterations=200)


def plot_power_flow(
    result: dict,
    outfile: str | Path = "case9_result.png",
    title: str = "IEEE 9-bus power flow",
) -> Path:
    """绘制线路负载率、母线类型和电压幅值。"""

    bus = result["bus"]
    branch = result["branch"]
    bus_numbers = bus[:, BUS_I].astype(int)
    positions = _layout(branch, bus_numbers)

    fig, (ax_net, ax_voltage) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.4),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    segments: list[list[tuple[float, float]]] = []
    loadings: list[float] = []
    for row in branch:
        if row[BR_STATUS] <= 0:
            continue
        f = int(row[F_BUS])
        t = int(row[T_BUS])
        if f not in positions or t not in positions:
            continue
        segments.append([positions[f], positions[t]])
        rating = float(row[RATE_A])
        flow = max(abs(float(row[PF])), abs(float(row[PT])))
        loadings.append(np.nan if rating <= 0 else 100.0 * flow / rating)

    valid = [value for value in loadings if np.isfinite(value)]
    scale = max(valid + [1.0])
    line_collection = LineCollection(
        segments,
        cmap="turbo",
        linewidths=[1.2 + 3.0 * min(value, scale) / scale for value in loadings],
        array=np.asarray(loadings),
        clim=(0, scale),
    )
    ax_net.add_collection(line_collection)

    type_style = {
        REF: ("s", "#d62728", "Slack"),
        PV: ("^", "#2ca02c", "PV generator"),
        PQ: ("o", "#1f77b4", "PQ load"),
    }
    for bus_type, (marker, color, label) in type_style.items():
        indices = np.flatnonzero(bus[:, BUS_TYPE].astype(int) == bus_type)
        ax_net.scatter(
            [positions[int(bus_numbers[i])][0] for i in indices],
            [positions[int(bus_numbers[i])][1] for i in indices],
            s=125,
            marker=marker,
            c=color,
            edgecolors="black",
            linewidths=1.0,
            zorder=3,
            label=label,
        )

    for i, number in enumerate(bus_numbers):
        x, y = positions[int(number)]
        ax_net.text(x + 0.025, y + 0.025, str(number), weight="bold", fontsize=10)
    ax_net.set_title(f"Network topology ({len(bus)} buses, {len(branch)} branches)")
    ax_net.set_aspect("equal")
    ax_net.axis("off")
    ax_net.legend(loc="lower center", ncol=3, frameon=False)
    colorbar = fig.colorbar(line_collection, ax=ax_net, orientation="horizontal", pad=0.05)
    colorbar.set_label("Line loading (%)")

    voltage = bus[:, VM]
    bars = ax_voltage.bar(bus_numbers, voltage, color="#3d8fd1", edgecolor="black")
    ax_voltage.axhline(1.0, color="red", linestyle="--", linewidth=1.2)
    ax_voltage.set_title("Bus voltage magnitude")
    ax_voltage.set_xlabel("Bus number")
    ax_voltage.set_ylabel("Voltage magnitude (p.u.)")
    ax_voltage.set_xticks(bus_numbers)
    ax_voltage.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, voltage):
        ax_voltage.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.4f}",
            ha="center",
            fontsize=8,
        )
    lower = min(0.94, float(voltage.min()) - 0.02)
    upper = max(1.04, float(voltage.max()) + 0.02)
    ax_voltage.set_ylim(lower, upper)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = Path(outfile).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
