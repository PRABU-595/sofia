# Sofia: Same Model, Different Reasons

## Quantifying Sampling-Induced Explanation Instability in GNNs

Graph Neural Networks (GNNs) are widely explained post-hoc to understand their structural reasoning. However, when GNNs are scaled to large graphs using neighbor sampling techniques (like `NeighborLoader`), the subset of neighbors seen during training fundamentally changes the learned representations.

**Sofia** explores the core hypothesis that GNNs trained with neighbor sampling produce structurally different explanations than identical GNNs trained with full-neighborhood access. Critically, this disagreement is not random noise; it is causally linked to whether a node's most influential neighbors were excluded during sampled training. This instability is further modulated by graph homophily and node degree.

---

## Installation

Ensure you have Python 3.9+ installed. To install the required dependencies and package, run:

```bash
pip install -e .
```

This will install PyTorch, PyTorch Geometric, Captum, and other required scientific libraries as defined in `requirements.txt`.

---

## Quick Start

To reproduce the core analysis on the **Cora** dataset with a single command:

```bash
python experiments/run_full_pipeline.py --dataset cora --config configs/cora.yaml
```

This will automatically:
1. Train a full-batch and sampled GraphSAGE.
2. Track which true neighbors were excluded during sampling.
3. Compute GNNExplainer importance masks for both models.
4. Calculate Jaccard agreement, Spearman correlation, and SOFIA-Index.
5. Generate figures and output summary statistics to `outputs/cora/` and `outputs/results_summary.csv`.

---

## Full Experiment Reproduction

To reproduce the complete study across all four datasets (Cora, CiteSeer, Chameleon, Actor), run:

```bash
python experiments/run_full_pipeline.py --dataset cora --config configs/cora.yaml
python experiments/run_full_pipeline.py --dataset citeseer --config configs/citeseer.yaml
python experiments/run_full_pipeline.py --dataset chameleon --config configs/chameleon.yaml
python experiments/run_full_pipeline.py --dataset actor --config configs/actor.yaml
```

You can view the final cross-dataset summary table by running:

```bash
python experiments/cross_dataset_summary.py
```

---

## Output File Description

For each dataset, outputs are saved in `outputs/{dataset}/`:

- `models/gnn_full.pt`: Weights for the full-batch trained model.
- `models/gnn_sampled.pt`: Weights for the sampling-trained model.
- `explanations/explanations_full.pkl`: Pickled dictionary mapping node ID to top-k explanation indices for the full model.
- `explanations/explanations_sampled.pkl`: Pickled dictionary for the sampled model.
- `metrics/agreement_report.csv`: Contains per-node Jaccard, Spearman correlation, and Overlap@k.
- `metrics/noise_floor.csv`: Baseline instability (Jaccard) when perturbing the full model with input noise.
- `metrics/sofia_index.csv`: Master dataframe containing the computed SOFIA-Index per node alongside node degree, homophily, and exclusion rate.
- `metrics/regression_results.txt`: Output of the multivariate OLS regression predicting SOFIA-Index from exclusion.
- `analysis/homophily_stratified.csv`: Mean SOFIA-Index per homophily bin.
- `analysis/degree_stratified.csv`: Mean SOFIA-Index per degree quintile.
- `figures/`: PDFs and PNGs containing KDE plots of instability, heatmap samples, and LOWESS curves for degree controls.

---

## Main Results Table Schema (`outputs/results_summary.csv`)

| Column Name | Description |
|---|---|
| Dataset | Name of the dataset (e.g., cora, chameleon) |
| Homophily | Mean label homophily of the dataset |
| Cross-Model Jaccard | Mean Jaccard agreement between full and sampled model explanations |
| Same-Model Jaccard (NF) | Mean baseline noise floor (Jaccard of same model under input perturbation) |
| SOFIA-Index (mean) | The computed mean structural instability index |
| beta_exclusion | The OLS coefficient linking influential neighbor exclusion to SOFIA-Index |
| p-value | The statistical significance of `beta_exclusion` |

---

