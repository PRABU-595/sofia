# Sofia: Quantifying Sampling-Induced Explanation Instability in Graph Neural Networks

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

## Datasets

This study evaluates three distinct benchmark datasets to control for homophily and degree:
1. **Cora** (Citation Network: High Homophily, Low Degree)
2. **Actor** (Film Co-occurrence: Heterophilous, Low Degree)
3. **Reddit** (Social Network: High Homophily, High Degree)

**Dataset Access:**
* **Cora and Actor** datasets are small and are included directly in this repository under `data/`. PyTorch Geometric will process them automatically.
* **Reddit** is extremely large and therefore not included in this repository. 
  * Download the raw Reddit zip file here: https://data.dgl.ai/dataset/reddit.zip
  * PyTorch Geometric will automatically download this file and extract it when you run the pipeline.

---

## Quick Start

To reproduce the core analysis on the **Cora** dataset with a single command:

```bash
python experiments/run_full_pipeline.py --dataset cora --config configs/cora.yaml --explainer saliency
```

This will automatically:
1. Train a full-batch and sampled GraphSAGE.
2. Track which true neighbors were excluded during sampling.
3. Compute explanation importance masks for both models.
4. Calculate Jaccard agreement, Spearman correlation, and SOFIA-Index.
5. Generate figures and output summary statistics to `outputs/cora/`.

---

## Full Experiment Reproduction

To reproduce the complete study across all three final datasets (Cora, Actor, Reddit), run:

```bash
python experiments/run_full_pipeline.py --dataset cora --config configs/cora.yaml --explainer saliency
python experiments/run_full_pipeline.py --dataset actor --config configs/actor_aggressive.yaml --explainer saliency
python experiments/run_full_pipeline.py --dataset reddit --config configs/reddit_aggressive.yaml --explainer saliency
```

*Note: The Reddit dataset requires significant memory and a GPU. The aggressive configurations simulate heavy mini-batch sampling restrictions.*

---

## Output File Description

For each dataset, outputs are saved in `outputs/{dataset}/`:

- `models/gnn_full.pt`: Weights for the full-batch trained model.
- `models/gnn_sampled_seed_X.pt`: Weights for the sampling-trained models (across multiple seeds).
- `explanations/explanations_full.pkl`: Pickled dictionary mapping node ID to top-k explanation indices for the full model.
- `explanations/exp_sampled_seed_X.pkl`: Pickled dictionary for the sampled models.
- `metrics/noise_floor.csv`: Baseline instability (Jaccard) when perturbing the full model with input noise.
- `metrics/sofia_index.csv`: Master dataframe containing the computed SOFIA-Index per node alongside node degree, homophily, and exclusion rate.
- `metrics/regression_results.txt`: Output of the multivariate OLS regression predicting SOFIA-Index from exclusion.
- `figures/`: PDFs and PNGs containing KDE plots of instability, heatmap samples, and LOWESS curves for degree controls.

---

## Main Results Table

A consolidated summary of all runs can be found in `results_summary/Three_Dataset_Comparison_Table.csv`.

| Column Name | Description |
|---|---|
| Dataset | Name of the dataset (e.g., cora, actor, reddit) |
| Homophily | Mean label homophily of the dataset |
| Cross-Model Jaccard | Mean Jaccard agreement between full and sampled model explanations |
| Same-Model Jaccard (NF) | Mean baseline noise floor (Jaccard of same model across random seeds) |
| SOFIA-Index (mean) | The computed mean structural instability index |
| beta_exclusion | The OLS coefficient linking influential neighbor exclusion to SOFIA-Index |
| p-value | The statistical significance of `beta_exclusion` |
