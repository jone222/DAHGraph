# DAHGraph: Main Experiments

This release contains the code and indexed inputs for the three main evaluation protocols reported in the paper:

- Random triplet split (`random_triplet`)
- Compositional split (`compositional_v3`)
- Strict pair-disjoint split (`pair_disjoint_v3`)

Triplets use integer node indices, and graph/features are supplied in the same index order.

## Requirements

Python 3.10+, PyTorch, PyTorch Geometric, NumPy, pandas, and scikit-learn. Install matching PyTorch and PyG builds for your system, then install the remaining packages with:

```bash
pip install -r requirements.txt
```

## Train

Run a single seed and split:

```bash
python -m scripts.train --split random_triplet --seed 42 --device cuda
```

The default training configuration matches the paper: hidden dimension 512, three hybrid graph layers, four HGT heads, four R-GCN bases, dropout 0.1, AdamW, learning rate `5e-5`, weight decay `1e-4`, batch size 1024, at most 200 epochs, early-stopping patience 15, and validation Macro-F1 model selection. Set `--device cpu` when CUDA is unavailable. Results and checkpoints are written under `runs/`.

To run all three splits with the five paper seeds:

```bash
for split in random_triplet compositional_v3 pair_disjoint_v3; do
  for seed in 1 42 815 1203 2026; do
    python -m scripts.train --split "$split" --seed "$seed" --device cuda
  done
done
```

## Data

`data/splits/` contains train/validation/test CSV files with `drug_index`, `microbe_index`, `disease_index`, `label_id`, and `sample_weight`. Label 0 is potentially beneficial; label 1 is potentially harmful. The graph files contain indexed relations. Feature tensors are precomputed numerical inputs aligned to the node indices.

The trained checkpoints are not included. Each archived checkpoint is approximately 220 MB, above GitHub's 100 MB per-file limit.

The strict pair-disjoint dataset contains the 94,644 triplets retained after applying the pair-separation constraints; it is not the full 230,370-triplet set. The random and compositional splits use all 230,370 labeled triplets.

Check the packaged row counts, label schema, and node-index alignment with:

```bash
python -m scripts.validate_data
```
