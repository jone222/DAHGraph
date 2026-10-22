from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import HGTConv

from .multitask_hgt import MultiTaskTripletHGT, PairDecoder
from .triplet_features import compose_triplet_features


def _mean_signed_messages(
    messages: torch.Tensor,
    dst_index: torch.Tensor,
    num_dst: int,
) -> torch.Tensor:
    out = messages.new_zeros((num_dst, messages.size(-1)))
    count = messages.new_zeros((num_dst, 1))
    if dst_index.numel() == 0:
        return out
    out.index_add_(0, dst_index.long(), messages)
    count.index_add_(0, dst_index.long(), torch.ones((dst_index.numel(), 1), device=messages.device, dtype=messages.dtype))
    return out / count.clamp_min(1.0)


class SignedPolarityMessageBlock(nn.Module):
    """Direction-aware residual messages for DM and MD signed relations."""

    SIGNED_RELATIONS = {
        ("drug", "increases", "microbe"): ("dm_positive", 1),
        ("drug", "decreases", "microbe"): ("dm_negative", -1),
        ("microbe", "increased_by", "drug"): ("dm_positive_reverse", 1),
        ("microbe", "decreased_by", "drug"): ("dm_negative_reverse", -1),
        ("microbe", "positively_associated_with", "disease"): ("md_positive", 1),
        ("microbe", "negatively_associated_with", "disease"): ("md_negative", -1),
        ("disease", "positively_associated_with", "microbe"): ("md_positive_reverse", 1),
        ("disease", "negatively_associated_with", "microbe"): ("md_negative_reverse", -1),
    }

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.rel_names = [name for name, _ in self.SIGNED_RELATIONS.values()]
        self.rel_index = {name: i for i, name in enumerate(self.rel_names)}
        self.rel_embed = nn.Embedding(len(self.rel_names), hidden_dim)
        self.src_proj = nn.ModuleDict({name: nn.Linear(hidden_dim, hidden_dim) for name in self.rel_names})
        self.attn = nn.ModuleDict({name: nn.Linear(hidden_dim * 3, 1) for name in self.rel_names})
        self.dst_proj = nn.ModuleDict(
            {
                "drug": nn.Linear(hidden_dim, hidden_dim),
                "microbe": nn.Linear(hidden_dim, hidden_dim),
                "disease": nn.Linear(hidden_dim, hidden_dim),
            }
        )
        self.dropout = nn.Dropout(dropout)
        self.last_relation_weight: dict[str, float] = {}

    def forward(self, h: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        signed_sum = {
            nt: h[nt].new_zeros(h[nt].shape)
            for nt in ["drug", "microbe", "disease"]
            if nt in h
        }
        relation_weights: dict[str, list[torch.Tensor]] = {}
        for etype, (rel_name, polarity) in self.SIGNED_RELATIONS.items():
            if etype not in edge_index_dict:
                continue
            src_type, _, dst_type = etype
            if src_type not in h or dst_type not in h:
                continue
            edge = edge_index_dict[etype]
            if edge.numel() == 0:
                continue
            src_idx, dst_idx = edge[0].long(), edge[1].long()
            rel = self.rel_embed.weight[self.rel_index[rel_name]].view(1, -1)
            src_h = h[src_type][src_idx]
            dst_h = h[dst_type][dst_idx]
            msg = self.src_proj[rel_name](src_h) + float(polarity) * rel
            gate = torch.sigmoid(self.attn[rel_name](torch.cat([src_h, dst_h, rel.expand_as(src_h)], dim=-1)))
            relation_weights.setdefault(rel_name, []).append(gate.detach().mean())
            msg = msg * gate
            signed_sum[dst_type] = signed_sum[dst_type] + _mean_signed_messages(msg, dst_idx, h[dst_type].size(0))

        out = {}
        for nt, x in h.items():
            if nt in signed_sum:
                out[nt] = x + self.dropout(torch.nn.functional.gelu(self.dst_proj[nt](signed_sum[nt])))
            else:
                out[nt] = x
        self.last_relation_weight = {
            name: float(torch.stack(vals).mean().cpu())
            for name, vals in relation_weights.items()
            if vals
        }
        return out


class HierarchicalTaxonomyEncoder(nn.Module):
    def __init__(self, vocab_sizes: dict[str, int], hidden_dim: int = 256, levels: list[str] | None = None) -> None:
        super().__init__()
        default_levels = ["phylum", "class", "order", "family", "genus", "species"]
        self.levels = [level for level in (levels or default_levels) if level in vocab_sizes]
        embed_dim = max(32, hidden_dim // 4)
        self.embeddings = nn.ModuleDict(
            {level: nn.Embedding(int(vocab_sizes[level]), embed_dim, padding_idx=0) for level in self.levels}
        )
        self.level_pos = nn.Parameter(torch.randn(len(self.levels), embed_dim) * 0.02)
        self.attn = nn.Linear(embed_dim, 1)
        self.proj = nn.Sequential(nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.last_taxonomy_attention: torch.Tensor | None = None

    def forward(self, taxonomy_ids: dict[str, torch.Tensor], taxonomy_mask: torch.Tensor) -> torch.Tensor:
        if not self.levels:
            raise RuntimeError("no_taxonomy_levels_available")
        parts = []
        masks = []
        for i, level in enumerate(self.levels):
            parts.append(self.embeddings[level](taxonomy_ids[level].long()) + self.level_pos[i].view(1, -1))
            if isinstance(taxonomy_mask, dict):
                masks.append(taxonomy_mask[level].float())
            else:
                masks.append(taxonomy_mask[:, i].float())
        stacked = torch.stack(parts, dim=1)
        mask = torch.stack(masks, dim=1)
        logits = self.attn(stacked).squeeze(-1).masked_fill(mask <= 0, -1e9)
        empty = mask.sum(dim=1, keepdim=True) <= 0
        logits = torch.where(empty, torch.zeros_like(logits), logits)
        weights = torch.softmax(logits, dim=1) * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        self.last_taxonomy_attention = weights.detach()
        return self.proj((stacked * weights.unsqueeze(-1)).sum(dim=1))


class DirectionalHypergraphBranch(nn.Module):
    """Candidate triplet hyperedge encoder with role-aware tokens."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.role_proj = nn.ModuleDict(
            {
                "drug": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)),
                "microbe": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)),
                "disease": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)),
            }
        )
        self.pair_proj = nn.ModuleDict(
            {
                "dm": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)),
                "md": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)),
                "dir": nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)),
            }
        )
        self.prior_proj = nn.Sequential(nn.Linear(5, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.seed_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.token_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.token_type = nn.Parameter(torch.randn(11, hidden_dim) * 0.02)
        self.attn = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.hyper_norm = nn.LayerNorm(hidden_dim)
        self.hyper_dropout = nn.Dropout(dropout)
        self.hyper_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        hd: torch.Tensor,
        hm: torch.Tensor,
        hs: torch.Tensor,
        z_dm: torch.Tensor,
        z_md: torch.Tensor,
        direction_h: torch.Tensor,
        priors: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seed = self.seed_proj(torch.cat([hd, hm, hs, z_dm, z_md, direction_h], dim=-1))
        priors = priors or {}
        drug_prior = priors.get("drug_prior", hd.new_zeros((hd.size(0), 5)))
        microbe_prior = priors.get("microbe_prior", hd.new_zeros((hd.size(0), 5)))
        disease_prior = priors.get("disease_prior", hd.new_zeros((hd.size(0), 5)))
        dm_pair_prior = priors.get("dm_pair_prior", hd.new_zeros((hd.size(0), 5)))
        md_pair_prior = priors.get("md_pair_prior", hd.new_zeros((hd.size(0), 5)))
        tokens = torch.stack(
            [
                self.role_proj["drug"](hd),
                self.role_proj["microbe"](hm),
                self.role_proj["disease"](hs),
                self.pair_proj["dm"](z_dm),
                self.pair_proj["md"](z_md),
                self.pair_proj["dir"](direction_h),
                self.prior_proj(drug_prior),
                self.prior_proj(microbe_prior),
                self.prior_proj(disease_prior),
                self.prior_proj(dm_pair_prior),
                self.prior_proj(md_pair_prior),
            ],
            dim=1,
        )
        tokens = self.token_proj(tokens) + self.token_type.view(1, 11, -1)
        seed_expand = seed.unsqueeze(1).expand_as(tokens)
        score = self.attn(torch.cat([seed_expand, tokens], dim=-1)).squeeze(-1)
        attn = torch.softmax(score, dim=1)
        self.last_attention = attn.detach()
        hyper_rep = self.hyper_norm(seed + self.hyper_dropout((attn.unsqueeze(-1) * tokens).sum(dim=1)))
        hyper_logits = self.hyper_decoder(hyper_rep)
        return hyper_rep, hyper_logits


class MechanismAwareEvidenceRouter(nn.Module):
    """Route local protein/pathway evidence for a candidate triplet."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.protein_proj = nn.Linear(hidden_dim, hidden_dim)
        self.pathway_proj = nn.Linear(hidden_dim, hidden_dim)
        self.protein_type = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.pathway_type = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.dm_query = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.md_query = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.null_protein = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.null_pathway = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _adjacency(edge_index_dict: dict, etypes: tuple[tuple[str, str, str], ...]) -> dict[int, set[int]]:
        result: dict[int, set[int]] = {}
        for etype in etypes:
            edge = edge_index_dict.get(etype)
            if edge is None or edge.numel() == 0:
                continue
            src, dst = edge[0].detach().cpu().long(), edge[1].detach().cpu().long()
            for a, b in zip(src.tolist(), dst.tolist()):
                result.setdefault(int(a), set()).add(int(b))
        return result

    def set_adjacency(self, edge_index_dict: dict) -> None:
        self.drug_protein = self._adjacency(
            edge_index_dict,
            (("drug", "response_affected_by_protein", "protein"),),
        )
        self.microbe_protein = self._adjacency(
            edge_index_dict,
            (("microbe", "related_to_protein", "protein"),),
        )
        self.microbe_pathway = self._adjacency(
            edge_index_dict,
            (("microbe", "associated_with_pathway", "pathway"),),
        )
        self.disease_microbe = self._adjacency(
            edge_index_dict,
            (
                ("disease", "positively_associated_with", "microbe"),
                ("disease", "negatively_associated_with", "microbe"),
            ),
        )

    @staticmethod
    def _unique_evidence(
        protein_ids: set[int],
        pathway_ids: set[int],
    ) -> list[tuple[str, int]]:
        return [
            *[("protein", int(i)) for i in sorted(protein_ids)],
            *[("pathway", int(i)) for i in sorted(pathway_ids)],
        ]

    def _route(
        self,
        query: torch.Tensor,
        evidence: list[tuple[str, int]],
        h: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not evidence:
            evidence = [("protein", -1)]
        tokens = []
        ids = []
        types = []
        for kind, index in evidence:
            if kind == "protein":
                token = self.null_protein if index < 0 else self.protein_proj(h["protein"][index]) + self.protein_type
                type_id = 0
            else:
                token = self.null_pathway if index < 0 else self.pathway_proj(h["pathway"][index]) + self.pathway_type
                type_id = 1
            tokens.append(token)
            ids.append(index)
            types.append(type_id)
        tokens_t = torch.stack(tokens, dim=0)
        keys = self.key(tokens_t)
        values = self.value(tokens_t)
        scores = torch.matmul(keys, query) / (keys.size(-1) ** 0.5)
        weights = torch.softmax(scores, dim=0)
        return (
            torch.sum(weights.unsqueeze(-1) * values, dim=0),
            weights,
            torch.tensor(ids, device=query.device, dtype=torch.long),
            torch.tensor(types, device=query.device, dtype=torch.long),
        )

    def _route_batch(
        self,
        queries: torch.Tensor,
        evidence_lists: list[list[tuple[str, int]]],
        h: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(items) for items in evidence_lists)
        device = queries.device
        batch_size = len(evidence_lists)
        ids = torch.full((batch_size, width), -1, device=device, dtype=torch.long)
        types = torch.zeros((batch_size, width), device=device, dtype=torch.long)
        mask = torch.zeros((batch_size, width), device=device, dtype=torch.bool)
        for row, items in enumerate(evidence_lists):
            for col, (kind, index) in enumerate(items):
                ids[row, col] = int(index)
                types[row, col] = 0 if kind == "protein" else 1
                mask[row, col] = True
        safe_ids = ids.clamp_min(0)
        protein_idx = safe_ids.clamp_max(max(int(h["protein"].size(0)) - 1, 0))
        pathway_idx = safe_ids.clamp_max(max(int(h["pathway"].size(0)) - 1, 0))
        protein_tokens = self.protein_proj(h["protein"][protein_idx]) + self.protein_type
        pathway_tokens = self.pathway_proj(h["pathway"][pathway_idx]) + self.pathway_type
        valid = mask & ids.ge(0)
        tokens = torch.where(types.unsqueeze(-1).eq(0), protein_tokens, pathway_tokens)
        null_tokens = torch.where(
            types.unsqueeze(-1).eq(0),
            self.null_protein.view(1, 1, -1),
            self.null_pathway.view(1, 1, -1),
        )
        tokens = torch.where(valid.unsqueeze(-1), tokens, null_tokens)
        keys = self.key(tokens)
        values = self.value(tokens)
        scores = torch.einsum("bwd,bd->bw", keys, queries) / (keys.size(-1) ** 0.5)
        scores = scores.masked_fill(~mask, -1.0e9)
        weights = torch.softmax(scores, dim=1)
        return (
            torch.sum(weights.unsqueeze(-1) * values, dim=1),
            weights,
            ids,
            types,
        )

    def forward(
        self,
        h: dict[str, torch.Tensor],
        hd: torch.Tensor,
        hm: torch.Tensor,
        hs: torch.Tensor,
        z_dm: torch.Tensor,
        z_md: torch.Tensor,
        drug_index: torch.Tensor,
        microbe_index: torch.Tensor,
        disease_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        drug_ids = drug_index.detach().cpu().tolist()
        microbe_ids = microbe_index.detach().cpu().tolist()
        disease_ids = disease_index.detach().cpu().tolist()
        dm_evidence_lists: list[list[tuple[str, int]]] = []
        md_evidence_lists: list[list[tuple[str, int]]] = []
        for drug_id, microbe_id, disease_id in zip(drug_ids, microbe_ids, disease_ids):
            drug_proteins = self.drug_protein.get(drug_id, set())
            microbe_proteins = self.microbe_protein.get(microbe_id, set())
            microbe_pathways = self.microbe_pathway.get(microbe_id, set())
            dm_items = self._unique_evidence(
                drug_proteins | microbe_proteins,
                microbe_pathways,
            )
            dm_evidence_lists.append(dm_items or [("protein", -1)])

            context_microbes = self.disease_microbe.get(disease_id, set())
            md_proteins = set(microbe_proteins)
            md_pathways = set(microbe_pathways)
            for context_microbe in context_microbes:
                md_proteins.update(self.microbe_protein.get(context_microbe, set()))
                md_pathways.update(self.microbe_pathway.get(context_microbe, set()))
            md_items = self._unique_evidence(md_proteins, md_pathways)
            md_evidence_lists.append(md_items or [("protein", -1)])

        dm_queries = self.dm_query(torch.cat([hd, hm, z_dm], dim=-1))
        md_queries = self.md_query(torch.cat([hm, hs, z_md], dim=-1))
        z_dm_mech, dm_attention, dm_ids, dm_types = self._route_batch(dm_queries, dm_evidence_lists, h)
        z_md_mech, md_attention, md_ids, md_types = self._route_batch(md_queries, md_evidence_lists, h)
        return {
            "z_mech_dm": z_dm_mech,
            "z_mech_md": z_md_mech,
            "z_mech": self.fusion(torch.cat([z_dm_mech, z_md_mech], dim=-1)),
            "mechanism_attention_dm": dm_attention,
            "mechanism_attention_md": md_attention,
            "mechanism_evidence_ids_dm": dm_ids,
            "mechanism_evidence_ids_md": md_ids,
            "mechanism_evidence_types_dm": dm_types,
            "mechanism_evidence_types_md": md_types,
            "mechanism_evidence_count_dm": torch.tensor(
                [len(x) for x in dm_evidence_lists], device=hd.device, dtype=torch.long
            ),
            "mechanism_evidence_count_md": torch.tensor(
                [len(x) for x in md_evidence_lists], device=hd.device, dtype=torch.long
            ),
        }


class MechanisticResidualFusion(nn.Module):
    """Fast relation-aware Protein/Pathway mean-max pooling with residual gating."""

    def __init__(self, hidden_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.protein_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.pathway_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.disease_proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim))
        self.dm_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.md_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.mech_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.mech_to_base = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self._adjacency_set = False
        self._edge_specs: dict[str, torch.Tensor | None] = {}

    def set_adjacency(self, edge_index_dict: dict) -> None:
        # Evidence is taken only from the strict message-passing graph supplied by the runner.
        self._edge_specs = {
            "drug_protein": edge_index_dict.get(("protein", "affects_response_to", "drug")),
            "microbe_protein": edge_index_dict.get(("protein", "belongs_to_microbe", "microbe")),
            "microbe_pathway": edge_index_dict.get(("pathway", "pathway_of_microbe", "microbe")),
        }
        self._adjacency_set = True

    @staticmethod
    def _mean_max(
        evidence_h: torch.Tensor,
        edge: torch.Tensor | None,
        num_dst: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_dim = evidence_h.size(1)
        mean = evidence_h.new_zeros((num_dst, hidden_dim))
        maximum = evidence_h.new_zeros((num_dst, hidden_dim))
        count = evidence_h.new_zeros((num_dst, 1))
        if edge is None or edge.numel() == 0:
            return mean, maximum, count
        src = edge[0].long()
        dst = edge[1].long()
        mean.index_add_(0, dst, evidence_h[src])
        count.index_add_(0, dst, evidence_h.new_ones((dst.numel(), 1)))
        mean = mean / count.clamp_min(1.0)
        maximum.fill_(-torch.inf)
        max_index = dst.view(-1, 1).expand(-1, hidden_dim)
        maximum.scatter_reduce_(0, max_index, evidence_h[src], reduce="amax", include_self=True)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        return mean, maximum, count

    @staticmethod
    def _batch_mean_max(
        evidence_h: torch.Tensor,
        edge: torch.Tensor | None,
        target_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate only evidence edges touching targets in the current batch."""
        batch_size = target_index.numel()
        hidden_dim = evidence_h.size(1)
        mean = evidence_h.new_zeros((batch_size, hidden_dim))
        maximum = evidence_h.new_zeros((batch_size, hidden_dim))
        count = evidence_h.new_zeros((batch_size, 1))
        if edge is None or edge.numel() == 0 or batch_size == 0:
            return mean, maximum, count

        unique_targets, inverse = torch.unique(target_index, sorted=True, return_inverse=True)
        dst = edge[1].long()
        selected = torch.searchsorted(unique_targets, dst)
        valid = (selected < unique_targets.numel()) & unique_targets[selected.clamp_max(unique_targets.numel() - 1)].eq(dst)
        if not valid.any():
            return mean, maximum, count

        src = edge[0].long()[valid]
        local_dst = selected[valid]
        local_count = evidence_h.new_zeros((unique_targets.numel(), 1))
        local_mean = evidence_h.new_zeros((unique_targets.numel(), hidden_dim))
        local_mean.index_add_(0, local_dst, evidence_h[src])
        local_count.index_add_(0, local_dst, evidence_h.new_ones((local_dst.numel(), 1)))
        local_mean = local_mean / local_count.clamp_min(1.0)

        local_max = evidence_h.new_full((unique_targets.numel(), hidden_dim), -torch.inf)
        max_index = local_dst.view(-1, 1).expand(-1, hidden_dim)
        local_max.scatter_reduce_(0, max_index, evidence_h[src], reduce="amax", include_self=True)
        local_max = torch.where(local_count > 0, local_max, torch.zeros_like(local_max))

        mean = local_mean[inverse]
        maximum = local_max[inverse]
        count = local_count[inverse]
        return mean, maximum, count

    def forward(
        self,
        h: dict[str, torch.Tensor],
        hd: torch.Tensor,
        hm: torch.Tensor,
        hs: torch.Tensor,
        r_base: torch.Tensor,
        drug_index: torch.Tensor,
        microbe_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        protein_h = self.protein_proj(h["protein"])
        pathway_h = self.pathway_proj(h["pathway"])
        drug_p_mean, drug_p_max, drug_p_count = self._batch_mean_max(
            protein_h, self._edge_specs.get("drug_protein"), drug_index
        )
        microbe_p_mean, microbe_p_max, microbe_p_count = self._batch_mean_max(
            protein_h, self._edge_specs.get("microbe_protein"), microbe_index
        )
        microbe_k_mean, microbe_k_max, microbe_k_count = self._batch_mean_max(
            pathway_h, self._edge_specs.get("microbe_pathway"), microbe_index
        )
        dm_mech = self.dm_encoder(
            torch.cat(
                [
                    drug_p_mean,
                    drug_p_max,
                    microbe_p_mean,
                    microbe_p_max,
                    microbe_k_mean,
                    microbe_k_max,
                ],
                dim=-1,
            )
        )
        md_mech = self.md_encoder(
            torch.cat(
                [
                    microbe_p_mean,
                    microbe_p_max,
                    microbe_k_mean,
                    microbe_k_max,
                    self.disease_proj(hs),
                ],
                dim=-1,
            )
        )
        z_mech = self.mech_encoder(torch.cat([dm_mech, md_mech], dim=-1))
        r_mech = self.mech_to_base(z_mech)
        evidence_count = torch.cat(
            [
                drug_p_count,
                microbe_p_count,
                microbe_k_count,
            ],
            dim=-1,
        )
        evidence_mask = evidence_count.sum(dim=-1, keepdim=True).gt(0).float()
        gate = evidence_mask * torch.sigmoid(
            self.gate(torch.cat([r_base, r_mech, evidence_count], dim=-1))
        )
        return {
            "z_mech_dm": dm_mech,
            "z_mech_md": md_mech,
            "z_mech": z_mech,
            "r_mech": r_mech,
            "mechanism_gate": gate,
            "mechanism_evidence_statistics": evidence_count,
            "h_mechanistic": r_base + gate * r_mech,
        }


class SignedDirectionalHGT(MultiTaskTripletHGT):
    _UNSIGNED_EDGE_TYPES = {
        ("drug", "increases", "microbe"): ("drug", "dm", "microbe"),
        ("drug", "decreases", "microbe"): ("drug", "dm", "microbe"),
        ("microbe", "increased_by", "drug"): ("microbe", "dm_reverse", "drug"),
        ("microbe", "decreased_by", "drug"): ("microbe", "dm_reverse", "drug"),
        ("microbe", "positively_associated_with", "disease"): ("microbe", "md", "disease"),
        ("microbe", "negatively_associated_with", "disease"): ("microbe", "md", "disease"),
        ("disease", "positively_associated_with", "microbe"): ("disease", "md_reverse", "microbe"),
        ("disease", "negatively_associated_with", "microbe"): ("disease", "md_reverse", "microbe"),
    }

    def __init__(
        self,
        metadata,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        use_signed_hgt: bool = True,
        use_dm_md_aux: bool = False,
        use_directional_composition: bool = False,
        use_taxonomy_encoder: bool = False,
        taxonomy_vocab_sizes: dict[str, int] | None = None,
        use_metapath_attention: bool = False,
        use_context_composition: bool = False,
        use_logic_prior: bool = False,
        use_disease_context: bool = False,
        use_dm_only: bool = False,
        use_md_relation: bool = True,
        context_fusion_mode: str = "residual",
        decoder_feature_mode: str = "current",
        use_deep_disease_context: bool = False,
        use_directional_hypergraph: bool = False,
        hypergraph_only: bool = False,
        use_mechanism_evidence_routing: bool = False,
        use_mechanistic_residual_fusion: bool = False,
        use_interaction_only_pair_repr: bool = False,
    ) -> None:
        super().__init__(
            metadata,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_residual=use_residual,
            use_layer_norm=use_layer_norm,
            use_deim=use_dm_md_aux,
            use_mbda=use_dm_md_aux,
            use_pppa=False,
            use_mppa=False,
        )
        self.use_signed_hgt = use_signed_hgt
        self.use_dm_md_aux = use_dm_md_aux
        self.use_directional_composition = use_directional_composition
        self.use_taxonomy_encoder = use_taxonomy_encoder
        self.use_metapath_attention = use_metapath_attention
        self.use_context_composition = bool(use_context_composition)
        self.use_logic_prior = use_logic_prior
        self.use_disease_context = use_disease_context
        self.use_dm_only = use_dm_only
        self.use_md_relation = bool(use_md_relation)
        self.use_deep_disease_context = bool(use_deep_disease_context)
        self.use_directional_hypergraph = bool(use_directional_hypergraph)
        self.hypergraph_only = bool(hypergraph_only)
        self.use_mechanism_evidence_routing = bool(use_mechanism_evidence_routing)
        self.use_mechanistic_residual_fusion = bool(use_mechanistic_residual_fusion)
        self.use_interaction_only_pair_repr = bool(use_interaction_only_pair_repr)
        if context_fusion_mode not in {"residual", "direct_concat"}:
            raise ValueError(f"unsupported context_fusion_mode: {context_fusion_mode}")
        self.context_fusion_mode = context_fusion_mode
        if decoder_feature_mode not in {
            "current",
            "raw_triplet",
            "raw_triplet_context",
            "raw_triplet_absdiffs",
            "no_context",
            "no_md",
            "mechanism_routing",
            "mechanistic_residual",
            "mechanistic_concat",
            "mechanistic_residual_no_pair_encoder",
        }:
            raise ValueError(f"unsupported decoder_feature_mode: {decoder_feature_mode}")
        self.decoder_feature_mode = decoder_feature_mode
        self.signed_blocks = nn.ModuleList([SignedPolarityMessageBlock(hidden_dim, dropout) for _ in range(num_layers)]) if use_signed_hgt else nn.ModuleList()
        if not use_signed_hgt:
            node_types, edge_types = metadata
            unsigned_edges = []
            for edge_type in edge_types:
                unsigned_edges.append(self._UNSIGNED_EDGE_TYPES.get(edge_type, edge_type))
            unsigned_metadata = (node_types, list(dict.fromkeys(unsigned_edges)))
            self.convs = nn.ModuleList(
                [HGTConv(hidden_dim, hidden_dim, unsigned_metadata, heads=num_heads) for _ in range(num_layers)]
            )
        self.taxonomy_encoder = HierarchicalTaxonomyEncoder(taxonomy_vocab_sizes or {}, hidden_dim) if use_taxonomy_encoder else None
        self.taxonomy_fusion = nn.Sequential(nn.Linear(hidden_dim * 2 + 1, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)) if use_taxonomy_encoder else None

        self.dm_decoder = PairDecoder(hidden_dim, dropout)
        self.md_decoder = PairDecoder(hidden_dim, dropout)
        self.dm_only_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        if use_directional_composition:
            pair_input_dim = hidden_dim * (2 if self.use_interaction_only_pair_repr else 4)
            self.dm_repr = nn.Sequential(nn.Linear(pair_input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
            self.md_repr = nn.Sequential(nn.Linear(pair_input_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
            self.direction_embed = nn.Linear(4, hidden_dim)
            if self.decoder_feature_mode == "raw_triplet":
                compose_input_dim = hidden_dim * 3
            elif self.decoder_feature_mode == "raw_triplet_context":
                compose_input_dim = hidden_dim * 4
            elif self.decoder_feature_mode == "raw_triplet_absdiffs":
                compose_input_dim = hidden_dim * 6
            elif self.decoder_feature_mode == "no_context":
                compose_input_dim = hidden_dim * 5
            elif self.decoder_feature_mode == "no_md":
                compose_input_dim = hidden_dim * 4
            elif self.decoder_feature_mode == "mechanism_routing":
                compose_input_dim = hidden_dim * 6
            elif self.decoder_feature_mode in {"mechanistic_residual", "mechanistic_residual_no_pair_encoder"}:
                compose_input_dim = hidden_dim
            elif self.decoder_feature_mode == "mechanistic_concat":
                compose_input_dim = hidden_dim * 2
            else:
                compose_input_dim = hidden_dim * (
                    6 if use_context_composition and context_fusion_mode == "direct_concat" else 5
                )
            self.compose_decoder = nn.Sequential(
                nn.Linear(compose_input_dim + (2 if use_logic_prior else 0), hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            self.compose_feature_dim = compose_input_dim + (2 if use_logic_prior else 0)
            if self.decoder_feature_mode in {"mechanistic_residual", "mechanistic_concat", "mechanistic_residual_no_pair_encoder"}:
                self.mechanism_base_proj = nn.Sequential(
                    nn.Linear(
                        hidden_dim * (3 if self.decoder_feature_mode == "mechanistic_residual_no_pair_encoder" else 5),
                        hidden_dim,
                    ),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                )
            if use_context_composition:
                self.context_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.context_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.context_value = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.context_proj = nn.Linear(hidden_dim, hidden_dim)
                self.context_norm = nn.LayerNorm(hidden_dim)
                self.context_dropout = nn.Dropout(dropout)
            if self.use_deep_disease_context:
                self.disease_relation_gate = nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Sigmoid(),
                )
                self.disease_relation_dm_proj = nn.Linear(hidden_dim, hidden_dim)
                self.disease_relation_md_proj = nn.Linear(hidden_dim, hidden_dim)
                self.disease_relation_dm_norm = nn.LayerNorm(hidden_dim)
                self.disease_relation_md_norm = nn.LayerNorm(hidden_dim)
        if use_metapath_attention:
            self.path_encoders = nn.ModuleList(
                [
                    nn.Sequential(nn.Linear(hidden_dim * 8, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
                    for _ in range(3)
                ]
            )
            self.path_attention = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            self.metapath_decoder = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim), nn.Dropout(dropout), nn.Linear(hidden_dim, 2))
        if self.use_directional_hypergraph:
            self.hypergraph_branch = DirectionalHypergraphBranch(hidden_dim, dropout)
            self.hypergraph_base_proj = nn.Sequential(
                nn.Linear(self.compose_feature_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            self.hypergraph_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
            self.hypergraph_fusion_norm = nn.LayerNorm(hidden_dim)
            self.hypergraph_fusion_decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
        self._taxonomy_payload: dict[str, torch.Tensor] | None = None
        self._last_relation_weight: dict[str, float] = {}
        self._last_path_attention: torch.Tensor | None = None
        self._last_dm_logits: torch.Tensor | None = None
        self._last_md_logits: torch.Tensor | None = None
        self._last_context_attention: torch.Tensor | None = None
        self._last_logic_prior: torch.Tensor | None = None
        self._last_hyper_attention: torch.Tensor | None = None
        self._evidence_adjacency_set = False
        self._last_mechanism_outputs: dict[str, torch.Tensor] | None = None
        if self.use_mechanism_evidence_routing:
            self.mechanism_router = MechanismAwareEvidenceRouter(hidden_dim, dropout)
        if self.use_mechanistic_residual_fusion or self.decoder_feature_mode in {
            "mechanistic_residual",
            "mechanistic_concat",
            "mechanistic_residual_no_pair_encoder",
        }:
            self.mechanistic_fusion = MechanisticResidualFusion(hidden_dim, dropout)

    def set_evidence_adjacency(self, edge_index_dict: dict) -> None:
        if self.use_mechanism_evidence_routing:
            self.mechanism_router.set_adjacency(edge_index_dict)
            self._evidence_adjacency_set = True
        if self.use_mechanistic_residual_fusion or self.decoder_feature_mode in {
            "mechanistic_residual",
            "mechanistic_concat",
            "mechanistic_residual_no_pair_encoder",
        }:
            self.mechanistic_fusion.set_adjacency(edge_index_dict)
            self._evidence_adjacency_set = True

    @staticmethod
    def _pair_features(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, a * b, torch.abs(a - b)], dim=-1)

    def _relation_pair_features(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.use_interaction_only_pair_repr:
            return torch.cat([a * b, torch.abs(a - b)], dim=-1)
        return self._pair_features(a, b)

    def set_taxonomy_payload(self, taxonomy_ids: dict[str, torch.Tensor], taxonomy_mask) -> None:
        self._taxonomy_payload = {"taxonomy_ids": taxonomy_ids, "taxonomy_mask": taxonomy_mask}

    def _inject_taxonomy(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.use_taxonomy_encoder or self.taxonomy_encoder is None or self.taxonomy_fusion is None:
            return x_dict
        if self._taxonomy_payload is None:
            raise RuntimeError("taxonomy_payload_not_set")
        tax_ids = {k: v.to(x_dict["microbe"].device) for k, v in self._taxonomy_payload["taxonomy_ids"].items()}
        raw_mask = self._taxonomy_payload["taxonomy_mask"]
        if isinstance(raw_mask, dict):
            tax_mask = {k: v.to(x_dict["microbe"].device) for k, v in raw_mask.items()}
            has_tax = torch.stack([tax_mask[level] for level in self.taxonomy_encoder.levels], dim=1).sum(dim=1, keepdim=True).gt(0).float()
        else:
            tax_mask = raw_mask.to(x_dict["microbe"].device)
            has_tax = tax_mask.sum(dim=1, keepdim=True).gt(0).float()
        tax_h = self.taxonomy_encoder(tax_ids, tax_mask)
        out = dict(x_dict)
        out["microbe"] = self.taxonomy_fusion(torch.cat([x_dict["microbe"], tax_h, has_tax], dim=-1))
        return out

    def encode(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        h = self._inject_taxonomy(x_dict)
        if not self.use_signed_hgt:
            unsigned_edges: dict[tuple[str, str, str], list[torch.Tensor]] = {}
            for edge_type, edge_index in edge_index_dict.items():
                unsigned_type = self._UNSIGNED_EDGE_TYPES.get(edge_type, edge_type)
                unsigned_edges.setdefault(unsigned_type, []).append(edge_index)
            edge_index_dict = {
                edge_type: torch.cat(edge_indices, dim=1)
                for edge_type, edge_indices in unsigned_edges.items()
            }
        relation_weights = {}
        for i, conv in enumerate(self.convs):
            out = conv(h, edge_index_dict)
            if self.use_signed_hgt:
                out = self.signed_blocks[i]({nt: out.get(nt, x) for nt, x in h.items()}, edge_index_dict)
                relation_weights.update(self.signed_blocks[i].last_relation_weight)
            next_h = {}
            for nt, x in h.items():
                y = out.get(nt, x)
                if self.use_residual and y.shape == x.shape:
                    y = y + x
                if self.use_layer_norm:
                    y = self.norms[i][nt](y)
                next_h[nt] = self.dropout(torch.nn.functional.gelu(y))
            h = next_h
        self._last_relation_weight = relation_weights
        return h

    def deim_logits(self, h: dict[str, torch.Tensor], drug_index: torch.Tensor, microbe_index: torch.Tensor) -> torch.Tensor:
        return self.dm_decoder(h["drug"][drug_index], h["microbe"][microbe_index])

    def mbda_logits(self, h: dict[str, torch.Tensor], microbe_index: torch.Tensor, disease_index: torch.Tensor) -> torch.Tensor:
        return self.md_decoder(h["microbe"][microbe_index], h["disease"][disease_index])

    def triplet_representation(
        self,
        h: dict[str, torch.Tensor],
        drug_index: torch.Tensor,
        microbe_index: torch.Tensor,
        disease_index: torch.Tensor,
    ) -> torch.Tensor:
        hd = h["drug"][drug_index]
        hm = h["microbe"][microbe_index]
        hs = h["disease"][disease_index]
        if not self.use_directional_composition:
            return compose_triplet_features(hd, hm, hs)

        dm_logits = self.dm_decoder(hd, hm)
        md_logits = self.md_decoder(hm, hs) if self.use_md_relation else hd.new_zeros((hd.size(0), 2))
        z_dm = self.dm_repr(self._relation_pair_features(hd, hm))
        z_md = self.md_repr(self._relation_pair_features(hm, hs))
        if self.use_deep_disease_context:
            relation_gate = self.disease_relation_gate(torch.cat([hs, z_dm, z_md], dim=-1))
            z_dm = self.disease_relation_dm_norm(
                z_dm + relation_gate * self.disease_relation_dm_proj(z_dm)
            )
            z_md = self.disease_relation_md_norm(
                z_md + relation_gate * self.disease_relation_md_proj(z_md)
            )
        dm_prob = torch.softmax(dm_logits, dim=1)
        md_prob = torch.softmax(md_logits, dim=1) if self.use_md_relation else hd.new_zeros((hd.size(0), 2))
        direction_logic = torch.cat(
            [
                dm_prob[:, 0:1] * md_prob[:, 1:2],
                dm_prob[:, 1:2] * md_prob[:, 0:1],
                dm_prob[:, 0:1] * md_prob[:, 0:1],
                dm_prob[:, 1:2] * md_prob[:, 1:2],
            ],
            dim=-1,
        )
        direction_h = self.direction_embed(direction_logic)
        return torch.cat([hd, hm, hs, z_dm * z_md, direction_h], dim=-1)

    def _metapath_context(self, h: dict[str, torch.Tensor], edge_index_dict: dict, microbe_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hm = h["microbe"][microbe_index]
        protein_ctx_all = hm.new_zeros(h["microbe"].shape)
        pathway_ctx_all = hm.new_zeros(h["microbe"].shape)
        etype = ("protein", "belongs_to_microbe", "microbe")
        if etype in edge_index_dict and "protein" in h:
            edge = edge_index_dict[etype]
            protein_ctx_all = _mean_signed_messages(h["protein"][edge[0].long()], edge[1].long(), h["microbe"].size(0))
        etype = ("pathway", "pathway_of_microbe", "microbe")
        if etype in edge_index_dict and "pathway" in h:
            edge = edge_index_dict[etype]
            pathway_ctx_all = _mean_signed_messages(h["pathway"][edge[0].long()], edge[1].long(), h["microbe"].size(0))
        return protein_ctx_all[microbe_index], pathway_ctx_all[microbe_index]

    def main_logits(
        self,
        h: dict[str, torch.Tensor],
        drug_index: torch.Tensor,
        microbe_index: torch.Tensor,
        disease_index: torch.Tensor,
        edge_index_dict: dict | None = None,
        hypergraph_context: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        return self.main_outputs(h, drug_index, microbe_index, disease_index, edge_index_dict, hypergraph_context)["logits"]

    def main_outputs(
        self,
        h: dict[str, torch.Tensor],
        drug_index: torch.Tensor,
        microbe_index: torch.Tensor,
        disease_index: torch.Tensor,
        edge_index_dict: dict | None = None,
        hypergraph_context: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        hd = h["drug"][drug_index]
        hm = h["microbe"][microbe_index]
        dm_logits = self.dm_decoder(hd, hm)
        self._last_dm_logits = dm_logits.detach()
        dm_prob = torch.softmax(dm_logits, dim=1)

        if self.use_dm_only:
            z_dm = self.dm_repr(self._relation_pair_features(hd, hm)) if self.use_directional_composition else self._relation_pair_features(hd, hm)
            dm_only_logits = self.dm_only_decoder(
                torch.cat([hd, hm, z_dm, dm_prob], dim=-1)
            )
            self._last_md_logits = None
            self._last_context_attention = None
            self._last_logic_prior = None
            return {
                "logits": dm_only_logits,
                "final_logits": dm_only_logits,
                "dm_positive_probability": dm_prob[:, 0],
                "dm_negative_probability": dm_prob[:, 1],
                "md_positive_probability": dm_prob.new_zeros(dm_prob.size(0)),
                "md_negative_probability": dm_prob.new_zeros(dm_prob.size(0)),
                "beneficial_prior": dm_prob.new_zeros(dm_prob.size(0)),
                "harmful_prior": dm_prob.new_zeros(dm_prob.size(0)),
                "disease_context_attention": dm_prob.new_zeros((dm_prob.size(0), 2)),
                "z_dm": z_dm,
                "h_disease_contextualized": hd.new_zeros(hd.shape),
            }

        hs = h["disease"][disease_index]
        md_logits = self.md_decoder(hm, hs)
        self._last_md_logits = md_logits.detach()
        md_prob = torch.softmax(md_logits, dim=1)
        beneficial_prior = dm_prob[:, 0:1] * md_prob[:, 1:2] + dm_prob[:, 1:2] * md_prob[:, 0:1]
        harmful_prior = dm_prob[:, 0:1] * md_prob[:, 0:1] + dm_prob[:, 1:2] * md_prob[:, 1:2]
        logic_prior = torch.cat([beneficial_prior, harmful_prior], dim=-1)
        self._last_logic_prior = logic_prior.detach()
        if not self.use_directional_composition:
            logits = self.triplet_decoder(compose_triplet_features(hd, hm, hs))
            return {
                "logits": logits,
                "final_logits": logits,
                "dm_positive_probability": dm_prob[:, 0],
                "dm_negative_probability": dm_prob[:, 1],
                "md_positive_probability": md_prob[:, 0],
                "md_negative_probability": md_prob[:, 1],
                "beneficial_prior": beneficial_prior[:, 0],
                "harmful_prior": harmful_prior[:, 0],
                "disease_context_attention": hd.new_zeros((hd.size(0), 2)),
                "z_context": hd.new_zeros(hd.shape),
                "h_disease_contextualized": hs,
            }

        z_dm = self.dm_repr(self._relation_pair_features(hd, hm))
        z_md = self.md_repr(self._relation_pair_features(hm, hs)) if self.use_md_relation else hd.new_zeros(hd.shape)
        if self.use_md_relation:
            direction_logic = torch.cat(
                [
                    dm_prob[:, 0:1] * md_prob[:, 1:2],
                    dm_prob[:, 1:2] * md_prob[:, 0:1],
                    dm_prob[:, 0:1] * md_prob[:, 0:1],
                    dm_prob[:, 1:2] * md_prob[:, 1:2],
                ],
                dim=-1,
            )
        else:
            direction_logic = torch.cat(
                [dm_prob[:, 0:1], dm_prob[:, 1:2], hd.new_zeros((hd.size(0), 2))],
                dim=-1,
            )
        direction_h = self.direction_embed(direction_logic)
        context_attention = hd.new_zeros((hd.size(0), 2))
        z_context = hd.new_zeros(hd.shape)
        h_disease_contextualized = hs
        if self.use_context_composition:
            q = self.context_query(hs).unsqueeze(1)
            relation_tokens = torch.stack([z_dm, z_md], dim=1)
            k = self.context_key(relation_tokens)
            v = self.context_value(relation_tokens)
            scores = (q * k).sum(dim=-1) / (k.size(-1) ** 0.5)
            context_attention = torch.softmax(scores, dim=1)
            z_context = (context_attention.unsqueeze(-1) * v).sum(dim=1)
            h_disease_contextualized = self.context_norm(
                hs + self.context_dropout(self.context_proj(z_context))
            )
        self._last_context_attention = context_attention.detach()
        mechanism_outputs: dict[str, torch.Tensor] = {}
        z_mech = hd.new_zeros(hd.shape)
        if self.use_mechanism_evidence_routing:
            if not self._evidence_adjacency_set:
                self.set_evidence_adjacency(edge_index_dict or {})
            mechanism_outputs = self.mechanism_router(
                h,
                hd,
                hm,
                hs,
                z_dm,
                z_md,
                drug_index,
                microbe_index,
                disease_index,
            )
            z_mech = mechanism_outputs["z_mech"]
            self._last_mechanism_outputs = mechanism_outputs
        residual_mechanism_outputs: dict[str, torch.Tensor] = {}
        if self.decoder_feature_mode == "raw_triplet":
            fused_parts = [hd, hm, hs]
        elif self.decoder_feature_mode == "raw_triplet_context":
            fused_parts = [hd, hm, hs, z_context]
        elif self.decoder_feature_mode == "raw_triplet_absdiffs":
            fused_parts = [hd, hm, hs, torch.abs(hd - hm), torch.abs(hm - hs), torch.abs(hd - hs)]
        elif self.decoder_feature_mode == "no_context":
            fused_parts = [hd, hm, hs, z_dm, z_md]
        elif self.decoder_feature_mode == "no_md":
            fused_parts = [hd, hm, hs, z_dm]
        elif self.decoder_feature_mode == "mechanism_routing":
            fused_parts = [hd, hm, hs, z_dm, z_md, z_mech]
        elif self.decoder_feature_mode in {"mechanistic_residual", "mechanistic_concat", "mechanistic_residual_no_pair_encoder"}:
            if self.decoder_feature_mode == "mechanistic_residual_no_pair_encoder":
                base_features = torch.cat([hd, hm, hs], dim=-1)
            else:
                base_features = torch.cat([hd, hm, hs, z_dm, z_md], dim=-1)
            r_base = self.mechanism_base_proj(base_features)
            if not self._evidence_adjacency_set:
                self.set_evidence_adjacency(edge_index_dict or {})
            residual_mechanism_outputs = self.mechanistic_fusion(
                h, hd, hm, hs, r_base, drug_index, microbe_index
            )
            if self.decoder_feature_mode == "mechanistic_concat":
                evidence_mask = (
                    residual_mechanism_outputs["mechanism_evidence_statistics"]
                    .sum(dim=-1, keepdim=True)
                    .gt(0)
                    .to(r_base.dtype)
                )
                fused = torch.cat(
                    [r_base, evidence_mask * residual_mechanism_outputs["r_mech"]],
                    dim=-1,
                )
            else:
                fused = residual_mechanism_outputs["h_mechanistic"]
            logits = self.compose_decoder(
                torch.cat([fused, logic_prior], dim=-1) if self.use_logic_prior else fused
            )
            outputs = {
                "logits": logits,
                "final_logits": logits,
                "dm_positive_probability": dm_prob[:, 0],
                "dm_negative_probability": dm_prob[:, 1],
                "md_positive_probability": md_prob[:, 0],
                "md_negative_probability": md_prob[:, 1],
                "beneficial_prior": beneficial_prior[:, 0],
                "harmful_prior": harmful_prior[:, 0],
                "disease_context_attention": context_attention,
                "z_dm": z_dm,
                "z_md": z_md,
                "z_context": z_context,
                "h_disease_contextualized": h_disease_contextualized,
            }
            outputs.update(residual_mechanism_outputs)
            return outputs
        elif self.use_context_composition and self.context_fusion_mode == "direct_concat":
            fused_parts = [hd, hm, hs, z_dm * z_md, direction_h, z_context]
        else:
            fused_parts = [hd, hm, h_disease_contextualized, z_dm * z_md, direction_h]
        if self.use_logic_prior:
            fused_parts.append(logic_prior)
        fused = torch.cat(fused_parts, dim=-1)
        logits = self.compose_decoder(fused)
        hyper_rep = hd.new_zeros(hd.shape)
        hyper_logits = hd.new_zeros((hd.size(0), 2))
        hypergraph_gate = hd.new_zeros(hd.shape)
        if self.use_directional_hypergraph:
            hyper_rep, hyper_logits = self.hypergraph_branch(hd, hm, hs, z_dm, z_md, direction_h, hypergraph_context)
            self._last_hyper_attention = self.hypergraph_branch.last_attention
            if self.hypergraph_only:
                logits = hyper_logits
            else:
                baseline_rep = self.hypergraph_base_proj(fused)
                hypergraph_gate = self.hypergraph_gate(torch.cat([baseline_rep, hyper_rep], dim=-1))
                fused_rep = self.hypergraph_fusion_norm(baseline_rep + hypergraph_gate * hyper_rep)
                logits = self.hypergraph_fusion_decoder(fused_rep)
        else:
            self._last_hyper_attention = None
        if self.use_metapath_attention and edge_index_dict is not None:
            protein_ctx, pathway_ctx = self._metapath_context(h, edge_index_dict, microbe_index)
            path1 = self.path_encoders[0](compose_triplet_features(hd, hm, hs))
            path2 = self.path_encoders[1](torch.cat([hd, protein_ctx, hm, hs, hd * hm, protein_ctx * hm, hm * hs, torch.abs(protein_ctx - hm)], dim=-1))
            path3 = self.path_encoders[2](torch.cat([hd, pathway_ctx, hm, hs, hd * hm, pathway_ctx * hm, hm * hs, torch.abs(pathway_ctx - hm)], dim=-1))
            paths = torch.stack([path1, path2, path3], dim=1)
            attn = torch.softmax(self.path_attention(paths).squeeze(-1), dim=1)
            self._last_path_attention = attn.detach()
            path_h = (paths * attn.unsqueeze(-1)).sum(dim=1)
            logits = self.metapath_decoder(torch.cat([logits.new_zeros((logits.size(0), path_h.size(1))).add(path_h), direction_h], dim=-1))
        outputs = {
            "logits": logits,
            "final_logits": logits,
            "dm_positive_probability": dm_prob[:, 0],
            "dm_negative_probability": dm_prob[:, 1],
            "md_positive_probability": md_prob[:, 0],
            "md_negative_probability": md_prob[:, 1],
            "beneficial_prior": beneficial_prior[:, 0],
            "harmful_prior": harmful_prior[:, 0],
            "disease_context_attention": context_attention,
            "z_dm": z_dm,
            "z_md": z_md,
            "z_context": z_context,
            "h_disease_contextualized": h_disease_contextualized,
            "hypergraph_representation": hyper_rep,
            "hypergraph_logits": hyper_logits,
            "hypergraph_gate": hypergraph_gate,
            "hypergraph_attention": self._last_hyper_attention if self._last_hyper_attention is not None else hd.new_zeros((hd.size(0), 11)),
        }
        if self.use_mechanism_evidence_routing:
            outputs.update(mechanism_outputs)
        return outputs

    def relation_attention_summary(self) -> dict[str, float]:
        return dict(self._last_relation_weight)

    def path_attention_summary(self) -> dict[str, float]:
        if self._last_path_attention is None:
            return {}
        values = self._last_path_attention.mean(dim=0).detach().cpu().tolist()
        return {"path_direct_dms": float(values[0]), "path_protein": float(values[1]), "path_pathway": float(values[2])}
