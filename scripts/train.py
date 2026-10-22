from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.losses import weighted_cross_entropy
from src.metrics import compute_metrics
from src.relation_adaptive_hgt import RelationAdaptiveHGTEncoder
from src.signed_directional_hgt import SignedDirectionalHGT


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("random_triplet", "compositional_v3", "pair_disjoint_v3")
SEEDS = (1, 42, 815, 1203, 2026)
HIDDEN_DIM = 512


class Triplets(Dataset):
    def __init__(self, path: Path) -> None:
        self.frame = pd.read_csv(path)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index]
        return {
            "drug_index": torch.tensor(int(row.drug_index), dtype=torch.long),
            "microbe_index": torch.tensor(int(row.microbe_index), dtype=torch.long),
            "disease_index": torch.tensor(int(row.disease_index), dtype=torch.long),
            "label": torch.tensor(int(row.label_id), dtype=torch.long),
            "weight": torch.tensor(float(row.sample_weight), dtype=torch.float32),
        }


class MicrobeTextAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM * 2 + 1),
            nn.Linear(HIDDEN_DIM * 2 + 1, HIDDEN_DIM),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_DIM),
        )

    def forward(self, taxonomy: torch.Tensor, text: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([taxonomy, text, mask], dim=-1))


class DAHGraph(SignedDirectionalHGT):
    def __init__(self, graph: torch.nn.Module) -> None:
        super().__init__(
            graph.metadata(),
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            use_residual=True,
            use_layer_norm=True,
            use_signed_hgt=False,
            use_dm_md_aux=False,
            use_directional_composition=True,
            use_context_composition=False,
            use_logic_prior=False,
            use_disease_context=False,
            use_md_relation=True,
            decoder_feature_mode="mechanistic_residual",
            use_mechanistic_residual_fusion=True,
        )
        del self.convs
        del self.norms
        del self.signed_blocks
        self.hybrid_encoder = RelationAdaptiveHGTEncoder(
            graph.metadata(),
            hidden_dim=HIDDEN_DIM,
            num_layers=3,
            num_heads=4,
            dropout=0.1,
            num_bases=4,
        )

    def encode(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        return self.hybrid_encoder(self._inject_taxonomy(x_dict), edge_index_dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_graph(graph: torch.nn.Module, device: torch.device):
    graph = graph.to(device)
    return graph.edge_index_dict


def evaluate(model, loader, x_dict, edges, device):
    model.eval()
    with torch.no_grad():
        h = model.encode(x_dict, edges)
        labels, probabilities = [], []
        for batch in loader:
            logits = model.main_logits(
                h,
                batch["drug_index"].to(device),
                batch["microbe_index"].to(device),
                batch["disease_index"].to(device),
                edges,
            )
            probabilities.append(torch.softmax(logits, dim=1).cpu().numpy())
            labels.extend(batch["label"].numpy().tolist())
    return compute_metrics(np.asarray(labels), np.concatenate(probabilities, axis=0))


def run(split: str, seed: int, device_name: str, smoke: bool, output_root: Path) -> dict:
    set_seed(seed)
    device = torch.device(device_name)
    data = ROOT / "data"
    graph = torch.load(data / "graphs" / f"{split}.pt", map_location="cpu", weights_only=False)
    features = torch.load(data / "features" / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
    text_package = torch.load(data / "features" / "microbe_text_512.pt", map_location="cpu", weights_only=True)

    # Restore the RNG point immediately after the original feature preparation.
    torch.set_rng_state(features["rng_state"])
    x_dict = {key: value.to(device) for key, value in features["x_dict"].items()}
    text = text_package["features"].to(device)
    mask = text_package["mask"].to(device)
    edges = move_graph(graph, device)
    model = DAHGraph(graph).to(device)
    adapter = MicrobeTextAdapter().to(device)
    model.set_evidence_adjacency(edges)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(adapter.parameters()),
        lr=5e-5,
        weight_decay=1e-4,
    )

    split_dir = data / "splits" / split
    loaders = {
        name: DataLoader(
            Triplets(split_dir / f"{name}.csv"),
            batch_size=1024,
            shuffle=name == "train",
        )
        for name in ("train", "valid", "test")
    }
    epochs = 1 if smoke else 200
    best_f1, best_state, best_epoch, patience = -1.0, None, 0, 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        adapter.train()
        for batch in loaders["train"]:
            optimizer.zero_grad()
            x = dict(x_dict)
            x["microbe"] = adapter(x_dict["microbe"], text, mask)
            h = model.encode(x, edges)
            logits = model.main_logits(
                h,
                batch["drug_index"].to(device),
                batch["microbe_index"].to(device),
                batch["disease_index"].to(device),
                edges,
            )
            loss = weighted_cross_entropy(
                logits,
                batch["label"].to(device),
                batch["weight"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(adapter.parameters()), 5.0
            )
            optimizer.step()

        x = dict(x_dict)
        x["microbe"] = adapter(x_dict["microbe"], text, mask)
        valid = evaluate(model, loaders["valid"], x, edges, device)
        history.append({"epoch": epoch, **valid})
        if valid["macro_f1"] > best_f1:
            best_f1 = valid["macro_f1"]
            best_epoch = epoch
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "adapter": {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()},
            }
            patience = 0
        else:
            patience += 1
            if patience >= 15 and not smoke:
                break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state["model"])
    adapter.load_state_dict(best_state["adapter"])
    x = dict(x_dict)
    x["microbe"] = adapter(x_dict["microbe"], text, mask)
    test = evaluate(model, loaders["test"], x, edges, device)
    out = output_root / split / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state["model"],
            "adapter_state_dict": best_state["adapter"],
            "split": split,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
        },
        out / "best_model.pt",
    )
    pd.DataFrame(history).to_csv(out / "validation_history.csv", index=False)
    result = {"split": split, "seed": seed, "best_epoch": best_epoch, **test}
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DAHGraph on a paper main split")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smoke", action="store_true", help="Run one epoch to verify the pipeline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    run(args.split, args.seed, args.device, args.smoke, args.output_dir.resolve())


if __name__ == "__main__":
    main()
