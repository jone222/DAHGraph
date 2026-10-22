"""Relation-adaptive HGT encoder used by the independent FullV1 experiment.

The layer combines two message aggregators at the same depth:

* a pure HGT multi-head attention message (without HGT's internal skip path),
* a relation-specific R-GCN message (without a root/self transform).

The only residual update is applied after their type-specific scalar gate has
fused the normalized messages.  This keeps the architecture a single hybrid
message-passing layer rather than two complete encoders joined at the end.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.nn import ParameterDict
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.nn.conv.hgt_conv import (
    HeteroDictLinear,
    HeteroLinear,
    construct_bipartite_edge_index,
)
from torch_geometric.nn.inits import ones
from torch_geometric.utils import softmax


Metadata = Tuple[List[str], List[Tuple[str, str, str]]]
NodeInput = Union[int, Dict[str, int]]


class PureHGTConv(MessagePassing):
    """HGT attention aggregation without the built-in input skip connection.

    PyG's :class:`HGTConv` mixes its attention output with the input through a
    learned type-specific skip parameter.  FullV1 owns the residual update at
    the hybrid-layer level, so this small local implementation keeps only the
    HGT message path and its normal relation/type projections.
    """

    def __init__(
        self,
        in_channels: NodeInput,
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        if out_channels % heads != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by heads ({heads})"
            )

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = list(metadata[0])
        self.edge_types = list(metadata[1])
        self.edge_types_map = {
            edge_type: i for i, edge_type in enumerate(self.edge_types)
        }
        self.dst_node_types = {edge_type[-1] for edge_type in self.edge_types}

        self.kqv_lin = HeteroDictLinear(
            self.in_channels,
            self.out_channels * 3,
        )
        self.out_lin = HeteroDictLinear(
            self.out_channels,
            self.out_channels,
            types=self.node_types,
        )

        dim = out_channels // heads
        num_type_relations = heads * len(self.edge_types)
        self.k_rel = HeteroLinear(
            dim,
            dim,
            num_type_relations,
            bias=False,
            is_sorted=True,
        )
        self.v_rel = HeteroLinear(
            dim,
            dim,
            num_type_relations,
            bias=False,
            is_sorted=True,
        )
        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            self.p_rel["__".join(edge_type)] = Parameter(torch.empty(1, heads))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.p_rel)

    @staticmethod
    def _cat(
        x_dict: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, int]]:
        offset: Dict[str, int] = {}
        chunks: list[Tensor] = []
        start = 0
        for node_type, x in x_dict.items():
            chunks.append(x)
            offset[node_type] = start
            start += x.size(0)
        if not chunks:
            raise ValueError("HGT received an empty node feature dictionary")
        return torch.cat(chunks, dim=0), offset

    def _construct_src_node_feat(
        self,
        k_dict: Dict[str, Tensor],
        v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Tuple[Tensor, Tensor, Dict[Tuple[str, str, str], int]]:
        start = 0
        num_edge_types = len(self.edge_types)
        heads = self.heads
        dim = self.out_channels // heads
        keys: list[Tensor] = []
        values: list[Tensor] = []
        type_ids: list[Tensor] = []
        offsets: Dict[Tuple[str, str, str], int] = {}

        for edge_type in edge_index_dict:
            src_type = edge_type[0]
            node_count = k_dict[src_type].size(0)
            offsets[edge_type] = start
            start += node_count
            relation_offset = self.edge_types_map[edge_type]
            type_vec = (
                torch.arange(heads, device=k_dict[src_type].device)
                .view(-1, 1)
                .repeat(1, node_count)
                * num_edge_types
                + relation_offset
            )
            type_ids.append(type_vec)
            keys.append(k_dict[src_type])
            values.append(v_dict[src_type])

        key = torch.cat(keys, dim=0).transpose(0, 1).reshape(-1, dim)
        value = torch.cat(values, dim=0).transpose(0, 1).reshape(-1, dim)
        type_id = torch.cat(type_ids, dim=1).flatten()
        key = self.k_rel(key, type_id).view(heads, -1, dim).transpose(0, 1)
        value = self.v_rel(value, type_id).view(heads, -1, dim).transpose(0, 1)
        return key, value, offsets

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Optional[Tensor]]:
        kqv_dict = self.kqv_lin(x_dict)
        k_dict: Dict[str, Tensor] = {}
        q_dict: Dict[str, Tensor] = {}
        v_dict: Dict[str, Tensor] = {}
        dim = self.out_channels // self.heads
        for node_type, value in kqv_dict.items():
            key, query, value_part = torch.tensor_split(value, 3, dim=1)
            k_dict[node_type] = key.view(-1, self.heads, dim)
            q_dict[node_type] = query.view(-1, self.heads, dim)
            v_dict[node_type] = value_part.view(-1, self.heads, dim)

        query, dst_offset = self._cat(q_dict)
        key, value, src_offset = self._construct_src_node_feat(
            k_dict,
            v_dict,
            edge_index_dict,
        )
        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict,
            src_offset,
            dst_offset,
            edge_attr_dict=self.p_rel,
            num_nodes=key.size(0),
        )
        message = self.propagate(
            edge_index,
            k=key,
            q=query,
            v=value,
            edge_attr=edge_attr,
        )

        out_dict: Dict[str, Optional[Tensor]] = {}
        for node_type, start in dst_offset.items():
            end = start + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = message[start:end]

        projected = {
            node_type: torch.nn.functional.gelu(value)
            for node_type, value in out_dict.items()
            if value is not None
        }
        projected = self.out_lin(projected)
        out_dict = {
            node_type: projected.get(node_type)
            for node_type in out_dict
        }
        return out_dict

    def message(
        self,
        k_j: Tensor,
        q_i: Tensor,
        v_j: Tensor,
        edge_attr: Tensor,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        alpha = alpha / (q_i.size(-1) ** 0.5)
        alpha = softmax(alpha, index, ptr, size_i)
        return (v_j * alpha.view(-1, self.heads, 1)).view(-1, self.out_channels)


class RelationAdaptiveHGTLayer(nn.Module):
    """One HGT/R-GCN message layer with a type-specific scalar gate."""

    def __init__(
        self,
        metadata: Metadata,
        hidden_dim: int,
        heads: int,
        dropout: float,
        num_bases: int = 4,
        fixed_gate: float | None = None,
        fusion_mode: str = "gate",
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.node_types = list(node_types)
        self.edge_types = list(edge_types)
        self.relation_to_id = {
            edge_type: relation_id
            for relation_id, edge_type in enumerate(self.edge_types)
        }
        self.hgt = PureHGTConv(hidden_dim, hidden_dim, metadata, heads=heads)
        self.rgcn = RGCNConv(
            hidden_dim,
            hidden_dim,
            num_relations=len(self.edge_types),
            num_bases=num_bases,
            aggr="mean",
            root_weight=False,
        )
        self.hgt_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.rgcn_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.gates = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                )
                for node_type in self.node_types
            }
        )
        self.update_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.update_norm = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in self.node_types}
        )
        self.dropout = nn.Dropout(dropout)
        self.fixed_gate = fixed_gate
        if fusion_mode not in {"gate", "concat"}:
            raise ValueError(f"unsupported fusion_mode: {fusion_mode}")
        self.fusion_mode = fusion_mode
        self.concat_projection = (
            nn.Linear(hidden_dim * 2, hidden_dim)
            if fusion_mode == "concat"
            else None
        )
        self.last_gate: dict[str, Tensor] = {}
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for gate in self.gates.values():
            final = gate[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.bias)

    def _pack_edges(
        self,
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
        spans: Dict[str, Tuple[int, int]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        edges: list[Tensor] = []
        relation_ids: list[Tensor] = []
        for edge_type in self.edge_types:
            edge = edge_index_dict.get(edge_type)
            if edge is None or edge.numel() == 0:
                continue
            shifted = edge.long().to(device).clone()
            shifted[0] += spans[edge_type[0]][0]
            shifted[1] += spans[edge_type[2]][0]
            edges.append(shifted)
            relation_ids.append(
                torch.full(
                    (shifted.size(1),),
                    self.relation_to_id[edge_type],
                    dtype=torch.long,
                    device=device,
                )
            )
        if not edges:
            return (
                torch.empty((2, 0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.long, device=device),
            )
        return torch.cat(edges, dim=1), torch.cat(relation_ids, dim=0)

    def forward(
        self,
        h: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        hgt_messages = self.hgt(h, edge_index_dict)
        chunks: list[Tensor] = []
        spans: Dict[str, Tuple[int, int]] = {}
        start = 0
        for node_type in self.node_types:
            value = h[node_type]
            spans[node_type] = (start, start + value.size(0))
            chunks.append(value)
            start += value.size(0)
        packed = torch.cat(chunks, dim=0)
        edge_index, edge_type = self._pack_edges(
            edge_index_dict,
            spans,
            packed.device,
        )
        rgcn_packed = self.rgcn(packed, edge_index, edge_type)

        output: Dict[str, Tensor] = {}
        self.last_gate = {}
        for node_type in self.node_types:
            start, end = spans[node_type]
            h_current = h[node_type]
            hgt_message = hgt_messages.get(node_type)
            if hgt_message is None:
                hgt_message = torch.zeros_like(h_current)
            else:
                hgt_message = hgt_message.to(dtype=h_current.dtype)
            rgcn_message = rgcn_packed[start:end]
            hgt_message = self.hgt_norm[node_type](hgt_message)
            rgcn_message = self.rgcn_norm[node_type](rgcn_message)
            if self.fusion_mode == "concat":
                gate = torch.full(
                    (h_current.size(0), 1),
                    0.5,
                    dtype=h_current.dtype,
                    device=h_current.device,
                )
                fused_message = self.concat_projection(
                    torch.cat([hgt_message, rgcn_message], dim=-1)
                )
            else:
                gate_input = torch.cat([h_current, hgt_message, rgcn_message], dim=-1)
                if self.fixed_gate is None:
                    gate = torch.sigmoid(self.gates[node_type](gate_input))
                else:
                    gate = torch.full(
                        (h_current.size(0), 1),
                        float(self.fixed_gate),
                        dtype=h_current.dtype,
                        device=h_current.device,
                    )
                fused_message = gate * hgt_message + (1.0 - gate) * rgcn_message
            updated = h_current + self.dropout(self.update_ffn(fused_message))
            output[node_type] = self.update_norm[node_type](updated)
            self.last_gate[node_type] = gate.detach()
        return output


class RelationAdaptiveHGTEncoder(nn.Module):
    """Stacked relation-adaptive layers used as the FullV1 graph encoder."""

    def __init__(
        self,
        metadata: Metadata,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.2,
        num_bases: int = 4,
        fixed_gate: float | None = None,
        fusion_mode: str = "gate",
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                RelationAdaptiveHGTLayer(
                    metadata,
                    hidden_dim,
                    num_heads,
                    dropout,
                    num_bases=num_bases,
                    fixed_gate=fixed_gate,
                    fusion_mode=fusion_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], Tensor],
    ) -> Dict[str, Tensor]:
        h = x_dict
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h
