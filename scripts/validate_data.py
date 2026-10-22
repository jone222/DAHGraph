from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {
    "random_triplet": (184295, 23036, 23039),
    "compositional_v3": (184314, 23022, 23034),
    "pair_disjoint_v3": (75716, 9464, 9464),
}
EXPECTED_COLUMNS = [
    "drug_index",
    "microbe_index",
    "disease_index",
    "label_id",
    "sample_weight",
]


def main() -> None:
    for split, expected in EXPECTED_ROWS.items():
        sizes = []
        for part, expected_size in zip(("train", "valid", "test"), expected):
            path = ROOT / "data" / "splits" / split / f"{part}.csv"
            frame = pd.read_csv(path)
            if frame.columns.tolist() != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected columns in {path}: {frame.columns.tolist()}")
            if len(frame) != expected_size:
                raise ValueError(f"Unexpected row count in {path}: {len(frame)}")
            if not set(frame["label_id"].unique()).issubset({0, 1}):
                raise ValueError(f"Unexpected labels in {path}")
            if not frame["sample_weight"].gt(0).all():
                raise ValueError(f"Non-positive sample weight in {path}")
            bounds = {"drug_index": 885, "microbe_index": 434, "disease_index": 466}
            for column, count in bounds.items():
                if not frame[column].between(0, count - 1).all():
                    raise ValueError(f"Out-of-range {column} in {path}")
            sizes.append(len(frame))

        graph = torch.load(ROOT / "data" / "graphs" / f"{split}.pt", map_location="cpu", weights_only=False)
        expected_nodes = {"drug": 885, "microbe": 434, "disease": 466, "protein": 275, "pathway": 39}
        actual_nodes = {node_type: int(graph[node_type].num_nodes) for node_type in graph.node_types}
        if actual_nodes != expected_nodes:
            raise ValueError(f"Unexpected node counts for {split}: {actual_nodes}")
        print(f"{split}: rows={sizes}; nodes={actual_nodes}; relations={len(graph.edge_types)}")

    for seed in (1, 42, 815, 1203, 2026):
        features = torch.load(ROOT / "data" / "features" / f"seed_{seed}.pt", map_location="cpu", weights_only=True)
        counts = {key: int(value.shape[0]) for key, value in features["x_dict"].items()}
        if counts != {"drug": 885, "microbe": 434, "disease": 466, "protein": 275, "pathway": 39}:
            raise ValueError(f"Feature/node index mismatch for seed {seed}: {counts}")
    print("All packaged inputs passed validation.")


if __name__ == "__main__":
    main()
