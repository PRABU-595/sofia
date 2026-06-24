import argparse
import yaml
import os
import pickle
import pandas as pd
import torch
import random
import numpy as np
from collections import Counter

from models.gnn_full import GNNFull
from models.gnn_sampled import GNNSampled
from models.train import train_full, train_sampled, evaluate
from data.dataset_loader import DatasetLoader
from sampling.exclusion_tracker import ExclusionTracker
from explainers.explanation_runner import ExplanationRunner
from metrics.agreement import compute_agreement
from metrics.noise_floor import NoiseFloorEstimator
from metrics.instability_index import InstabilityIndex
from analysis.exclusion_correlation import ExclusionCorrelationAnalysis
from analysis.homophily_analysis import HomophilyStratifiedAnalysis
from analysis.degree_control import DegreeControlAnalysis
from visualization.explanation_heatmaps import plot_explanation_heatmaps
from visualization.agreement_distributions import plot_agreement_distributions
from visualization.instability_by_degree import plot_instability_by_degree

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fanout', type=int, nargs='+', default=None)
    parser.add_argument('--explainer', type=str, default='gnnexplainer')
    args = parser.parse_args()

    set_seed(args.seed)

    # 1. Load config
    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            dataset_config = yaml.safe_load(f)
            # Deep update
            for k, v in dataset_config.items():
                if k in config and isinstance(v, dict):
                    config[k].update(v)
                else:
                    config[k] = v

    if args.fanout:
        config['sampling']['fanout'] = args.fanout

    output_dir = os.path.join(config['experiment']['output_dir'], args.dataset)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'explanations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    # 1. Load dataset
    print(f"Loading {args.dataset}...")
    loader = DatasetLoader(root='./data')
    bundle = loader.load(args.dataset)
    data = bundle.data
    
    in_channels = data.x.shape[1]
    out_channels = int(data.y.max()) + 1
    
    # 2. Train GNNFull
    print("Training GNNFull...")
    set_seed(args.seed)
    model_full = GNNFull(
        in_channels, 
        config['model']['hidden_dim'], 
        out_channels, 
        num_layers=config['model']['num_layers'], 
        dropout=config['model']['dropout'],
        seed=args.seed
    )
    optimizer_full = torch.optim.Adam(model_full.parameters(), lr=config['model']['lr'])
    model_full = train_full(model_full, data, optimizer_full, epochs=config['model']['epochs'])
    acc_full = evaluate(model_full, data, data.test_mask)
    model_full.save(os.path.join(output_dir, 'models', 'gnn_full.pt'))
    
    # 3. Train GNNSampled & Initialize ExclusionTracker
    print("Training GNNSampled...")
    set_seed(args.seed)
    model_sampled = GNNSampled(
        in_channels, 
        config['model']['hidden_dim'], 
        out_channels, 
        num_layers=config['model']['num_layers'], 
        dropout=config['model']['dropout'],
        seed=args.seed
    )
    optimizer_sampled = torch.optim.Adam(model_sampled.parameters(), lr=config['model']['lr'])
    
    # 4. Initialize ExclusionTracker
    tracker = ExclusionTracker(num_nodes=data.num_nodes)
    
    model_sampled = train_sampled(
        model_sampled, data, optimizer_sampled, 
        fanout=config['sampling']['fanout'], 
        batch_size=config['sampling']['batch_size'],
        epochs=config['model']['epochs'],
        exclusion_tracker=tracker
    )
    acc_sampled = evaluate(model_sampled, data, data.test_mask)
    model_sampled.save(os.path.join(output_dir, 'models', 'gnn_sampled.pt'))
    
    print(f"Accuracy - Full: {acc_full:.4f}, Sampled: {acc_sampled:.4f}")

    # Prepare nodes to explain (test nodes with degree > 1)
    test_indices = data.test_mask.nonzero(as_tuple=True)[0].cpu().numpy()
    valid_indices = [idx for idx in test_indices if bundle.degree_per_node[idx] > 1]
    
    if len(valid_indices) > 500:
        np.random.seed(args.seed)
        valid_indices = np.random.choice(valid_indices, size=500, replace=False).tolist()
        
    print(f"Explaining {len(valid_indices)} nodes...")
    
    runner = ExplanationRunner(config['explainer'])
    
    # 5. Run ExplanationRunner on GNNFull
    exp_full = runner.run(model_full, data, valid_indices)
    with open(os.path.join(output_dir, 'explanations', 'explanations_full.pkl'), 'wb') as f:
        pickle.dump(exp_full, f)
        
    # 6. Run ExplanationRunner on GNNSampled
    exp_sampled = runner.run(model_sampled, data, valid_indices)
    with open(os.path.join(output_dir, 'explanations', 'explanations_sampled.pkl'), 'wb') as f:
        pickle.dump(exp_sampled, f)
        
    # 7. Compute NoiseFloor (Same-Model)
    print("Computing Noise Floor...")
    nf_estimator = NoiseFloorEstimator(runner, n_runs=config['metrics']['noise_floor_runs'])
    same_model_jaccard = nf_estimator.estimate(model_full, data, valid_indices)
    
    nf_df = pd.DataFrame(list(same_model_jaccard.items()), columns=['node_id', 'jaccard_same_model'])
    nf_df.to_csv(os.path.join(output_dir, 'metrics', 'noise_floor.csv'), index=False)
    
    # 8. Compute AgreementReport (Cross-Model)
    print("Computing Agreement Metrics...")
    agreement = compute_agreement(exp_full, exp_sampled)
    
    ag_df = pd.DataFrame({
        'node_id': list(agreement.per_node_jaccard.keys()),
        'jaccard_cross_model': list(agreement.per_node_jaccard.values()),
        'spearman': list(agreement.per_node_spearman.values())
    })
    ag_df.to_csv(os.path.join(output_dir, 'metrics', 'agreement_report.csv'), index=False)
    
    # 9. Compute SOFIA-Index
    print("Computing SOFIA-Index...")
    df_meta = tracker.get_dataframe(exp_full, bundle.degree_per_node, bundle.homophily_per_node)
    
    index_calculator = InstabilityIndex()
    df_sofia = index_calculator.compute(df_meta, agreement.per_node_jaccard, same_model_jaccard)
    df_sofia.to_csv(os.path.join(output_dir, 'metrics', 'sofia_index.csv'), index=False)
    
    # 10. Run ExclusionCorrelationAnalysis
    print("Running Regression Analysis...")
    analysis_corr = ExclusionCorrelationAnalysis()
    results_corr = analysis_corr.analyze(df_sofia)
    
    with open(os.path.join(output_dir, 'metrics', 'regression_results.txt'), 'w') as f:
        for k, v in results_corr.items():
            f.write(f"{k}: {v}\n")
            
    # 11. Run HomophilyStratifiedAnalysis
    analysis_hom = HomophilyStratifiedAnalysis()
    results_hom = analysis_hom.analyze(df_sofia)
    results_hom.to_csv(os.path.join(output_dir, 'analysis', 'homophily_stratified.csv'), index=False)
    
    # 12. Run DegreeControlAnalysis
    analysis_deg = DegreeControlAnalysis()
    results_deg = analysis_deg.analyze(df_sofia)
    results_deg.to_csv(os.path.join(output_dir, 'analysis', 'degree_stratified.csv'), index=False)
    
    # Visualizations
    print("Generating Figures...")
    nodes_to_plot = valid_indices[:3]
    plot_explanation_heatmaps(exp_full, exp_sampled, nodes_to_plot, os.path.join(output_dir, 'figures'), args.dataset)
    plot_agreement_distributions(df_sofia, os.path.join(output_dir, 'figures'), args.dataset)
    plot_instability_by_degree(df_sofia, os.path.join(output_dir, 'figures'), args.dataset)
    
    # 14. Print Summary Table
    mean_homophily = df_meta['homophily'][df_meta['homophily'] >= 0].mean()
    mean_cross_jaccard = agreement.mean_jaccard
    mean_same_jaccard = nf_df['jaccard_same_model'].mean()
    mean_sofia = df_sofia['sofia_index'].mean()
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(f"Dataset:            {args.dataset}")
    print(f"Homophily:          {mean_homophily:.4f}")
    print(f"Cross-Model Jaccard:{mean_cross_jaccard:.4f}")
    print(f"Same-Model Jaccard: {mean_same_jaccard:.4f}")
    print(f"SOFIA-Index (mean): {mean_sofia:.4f}")
    print(f"Beta Exclusion:     {results_corr['multivariate_coef_exclusion']:.4f}")
    print(f"p-value:            {results_corr['multivariate_p_value_exclusion']:.4e}")
    print("="*80)
    
    # Save a global summary line
    summary_line = f"{args.dataset},{mean_homophily:.2f},{mean_cross_jaccard:.4f},{mean_same_jaccard:.4f},{mean_sofia:.4f},{results_corr['multivariate_coef_exclusion']:.4f},{results_corr['multivariate_p_value_exclusion']:.4e}\n"
    with open('outputs/results_summary.csv', 'a') as f:
        f.write(summary_line)

if __name__ == "__main__":
    main()
