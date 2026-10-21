"""边感知图神经网络：从带掩码的节点量测预测全网电压和支路潮流。

模型只使用比赛允许的输入：

* 节点 ``[P, Q, V]`` 与三个掩码；
* 支路 ``[R, X, B/2]`` 与线路/变压器 one-hot；
* 图连接关系。

变压器变比、节点相角、真实净注入等辅助量都不进入前向计算。
第一版把线路和变压器的变比统一按 1 处理，模型直接学习官方标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .data import BRANCH_TARGET_NAMES, TARGET_NAMES


@dataclass(frozen=True)
class PowerFlowGNNConfig:
    """模型超参数。"""

    node_in: int = 6
    edge_in: int = 6
    hidden: int = 128
    num_layers: int = 6
    dropout: float = 0.1
    copy_known_v: bool = True


def _gather_nodes(node_embeddings: Tensor, index: Tensor) -> Tensor:
    """按 ``[B, E]`` 的节点下标从 ``[B, N, H]`` 中取特征。"""

    return torch.gather(
        node_embeddings,
        1,
        index.unsqueeze(-1).expand(-1, -1, node_embeddings.shape[-1]),
    )


class EdgeAwareLayer(nn.Module):
    """一层“边更新 + 节点聚合”的消息传递。

    每条边同时看首端、末端和边特征；消息聚合到末端节点后更新节点。
    图里显式加入反向边，因此一条物理支路上的信息可以双向流动。
    """

    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.hidden = hidden

        self.message = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.node_update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.edge_update = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )

        self.node_norm = nn.LayerNorm(hidden)
        self.edge_norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_h: Tensor,
        edge_h: Tensor,
        src: Tensor,
        dst: Tensor,
        node_valid: Tensor,
        edge_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, num_nodes, _ = node_h.shape
        num_edges = src.shape[1]

        h_src = _gather_nodes(node_h, src)
        h_dst = _gather_nodes(node_h, dst)

        messages = self.message(torch.cat([h_src, h_dst, edge_h], dim=-1))
        messages = messages * edge_valid.unsqueeze(-1)

        # 把 [B, E, H] 拉平成 [B*E, H]，按末端节点做 index_add。
        batch_offset = torch.arange(batch_size, device=node_h.device) * num_nodes
        dst_flat = (dst + batch_offset.unsqueeze(-1)).reshape(-1)
        agg = torch.zeros(
            batch_size * num_nodes, self.hidden, device=node_h.device, dtype=node_h.dtype
        )
        agg = agg.index_add(0, dst_flat, messages.reshape(-1, self.hidden))
        agg = agg.reshape(batch_size, num_nodes, self.hidden)

        degree = torch.zeros(
            batch_size * num_nodes, 1, device=node_h.device, dtype=node_h.dtype
        )
        degree = degree.index_add(
            0, dst_flat, edge_valid.reshape(-1, 1).to(dtype=node_h.dtype)
        )
        agg = agg / degree.reshape(batch_size, num_nodes, 1).clamp_min(1.0)

        node_h = self.node_norm(
            node_h + self.dropout(self.node_update(torch.cat([node_h, agg], dim=-1)))
        )
        node_h = node_h * node_valid.unsqueeze(-1)

        # 用更新后的节点状态再更新边状态，让支路头看到两端的最新信息。
        h_src = _gather_nodes(node_h, src)
        h_dst = _gather_nodes(node_h, dst)
        edge_h = self.edge_norm(
            edge_h
            + self.dropout(self.edge_update(torch.cat([edge_h, h_src, h_dst], dim=-1)))
        )
        edge_h = edge_h * edge_valid.unsqueeze(-1)

        return node_h, edge_h


class PowerFlowGNN(nn.Module):
    """输入一个带掩码的图，输出六个官方标签。"""

    def __init__(self, config: PowerFlowGNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or PowerFlowGNNConfig()

        self.node_encoder = nn.Sequential(
            nn.Linear(self.config.node_in, self.config.hidden),
            nn.LayerNorm(self.config.hidden),
            nn.GELU(),
        )
        # +1 是反向边标志：原始方向 0，反向 1。
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.config.edge_in + 1, self.config.hidden),
            nn.LayerNorm(self.config.hidden),
            nn.GELU(),
        )

        self.layers = nn.ModuleList(
            [
                EdgeAwareLayer(self.config.hidden, self.config.dropout)
                for _ in range(self.config.num_layers)
            ]
        )

        self.node_head = nn.Sequential(
            nn.LayerNorm(self.config.hidden),
            nn.Linear(self.config.hidden, self.config.hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden, 1),
        )
        self.branch_head = nn.Sequential(
            nn.LayerNorm(3 * self.config.hidden),
            nn.Linear(3 * self.config.hidden, self.config.hidden),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden, len(BRANCH_TARGET_NAMES)),
        )

        # 输出反标准化参数。训练时由 Normalizer 写入，随 checkpoint 保存。
        self.register_buffer("target_mean", torch.zeros(len(TARGET_NAMES)))
        self.register_buffer("target_std", torch.ones(len(TARGET_NAMES)))

    def set_target_stats(self, mean: Any, std: Any) -> None:
        """写入六个输出通道的均值和标准差。"""

        mean_tensor = torch.as_tensor(
            mean, dtype=self.target_mean.dtype, device=self.target_mean.device
        )
        std_tensor = torch.as_tensor(
            std, dtype=self.target_std.dtype, device=self.target_std.device
        )
        if mean_tensor.numel() != len(TARGET_NAMES) or std_tensor.numel() != len(TARGET_NAMES):
            raise ValueError(f"target stats 必须各有 {len(TARGET_NAMES)} 个值")
        with torch.no_grad():
            self.target_mean.copy_(mean_tensor)
            self.target_std.copy_(std_tensor)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """返回**标准化空间**的六个输出，形状见各键说明。"""

        node_x: Tensor = batch["node_x"]
        node_valid: Tensor = batch["node_valid"]
        edge_index: Tensor = batch["edge_index"]
        edge_attr: Tensor = batch["edge_attr"]
        edge_valid: Tensor = batch["edge_valid"]

        batch_size, num_nodes, _ = node_x.shape
        num_edges = edge_attr.shape[1]

        node_h = self.node_encoder(node_x) * node_valid.unsqueeze(-1)

        src = edge_index[..., 0]
        dst = edge_index[..., 1]

        # 原始边 + 反向边。反向边只改变消息方向，不改变物理支路参数。
        edge_attr_bidir = torch.cat([edge_attr, edge_attr], dim=1)
        reverse_flag = torch.cat(
            [
                torch.zeros_like(edge_attr[..., :1]),
                torch.ones_like(edge_attr[..., :1]),
            ],
            dim=1,
        )
        edge_h = self.edge_encoder(
            torch.cat([edge_attr_bidir, reverse_flag], dim=-1)
        )
        edge_valid_bidir = torch.cat([edge_valid, edge_valid], dim=1)
        edge_h = edge_h * edge_valid_bidir.unsqueeze(-1)

        src_bidir = torch.cat([src, dst], dim=1)
        dst_bidir = torch.cat([dst, src], dim=1)

        for layer in self.layers:
            node_h, edge_h = layer(
                node_h,
                edge_h,
                src_bidir,
                dst_bidir,
                node_valid,
                edge_valid_bidir,
            )

        # 支路输出只读原始方向的边，反向边不参与标签计算。
        h_src = _gather_nodes(node_h, src)
        h_dst = _gather_nodes(node_h, dst)
        branch_out = self.branch_head(
            torch.cat([h_src, h_dst, edge_h[:, :num_edges]], dim=-1)
        )

        vm = self.node_head(node_h).squeeze(-1)

        # 已知电压是送分题：输出阶段直接复制输入，不浪费模型容量。
        if self.config.copy_known_v and "known_v_mask" in batch:
            vm = torch.where(batch["known_v_mask"], batch["known_v_norm"], vm)

        outputs = {"vm_pu": vm}
        for index, name in enumerate(BRANCH_TARGET_NAMES):
            outputs[name] = branch_out[..., index]
        return outputs

    def forward_physical(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """返回反标准化后的物理量：Vm 为 p.u.，Δθ 为度，功率为 MW/MVar。"""

        normalized = self.forward(batch)
        return {
            name: normalized[name] * self.target_std[index] + self.target_mean[index]
            for index, name in enumerate(TARGET_NAMES)
        }

    @property
    def target_stats(self) -> tuple[Tensor, Tensor]:
        return self.target_mean, self.target_std
