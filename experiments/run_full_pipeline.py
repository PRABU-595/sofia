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

    with open('configs/base.yaml', 'r') as f:
        config = yaml.safe_load(f)
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            dataset_config = yaml.safe_load(f)
            for k, v in dataset_config.items():
                if k in config and isinstance(v, dict):
                    config[k].update(v)
                else:
                    config[k] = v

    if args.fanout:
        config['sampling']['fanout'] = args.fanout
    if args.explainer:
        config['explainer']['type'] = args.explainer

    output_dir = os.path.join(config['experiment']['output_dir'], args.dataset)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'explanations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'analysis'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)

    print(f"Loading {args.dataset}...")
    loader = DatasetLoader(root='./data')
    bundle = loader.load(args.dataset)
    data = bundle.data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
   
    in_channels = data.x.shape[1]
    out_channels = int(data.y.max()) + 1
   
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
    model_full = model_full.to(device)
    model_path = os.path.join(output_dir, 'models', 'gnn_full.pt')
    if os.path.exists(model_path):
        print(f"Loading existing GNNFull from {model_path}...")
        model_full.load_state_dict(torch.load(model_path))
        acc_full = evaluate(model_full, data, data.test_mask)
    else:
        optimizer_full = torch.optim.Adam(model_full.parameters(), lr=config['model']['lr'])
        model_full = train_full(model_full, data, optimizer_full, epochs=config['model']['epochs'])
        acc_full = evaluate(model_full, data, data.test_mask)
        torch.save(model_full.state_dict(), model_path)
   
    model_full.eval()
    pred_full = torch.zeros(data.num_nodes, dtype=torch.long, device=device)
    from torch_geometric.loader import NeighborLoader
    eval_loader = NeighborLoader(
        data,
        num_neighbors=[25, 10],
        batch_size=512,
        shuffle=False
    )
    with torch.no_grad():
        from tqdm import tqdm
        for batch in tqdm(eval_loader, desc="Getting Full Predictions"):

            batch = batch.to(device)
            out = model_full(batch.x, batch.edge_index)
            bs = batch.batch_size
            pred_full[batch.n_id[:bs]] = out[:bs].argmax(dim=1)

   
    # Select correctly classified nodes across all masks
    all_mask = data.train_mask | data.val_mask | data.test_mask
    correct_nodes = (pred_full == data.y) & all_mask
    correct_indices = correct_nodes.nonzero(as_tuple=True)[0].cpu().numpy()
   
    valid_indices = [idx for idx in correct_indices if bundle.degree_per_node[idx] > 1]
   
    max_nodes = config['experiment'].get('max_explanation_nodes', 1000)
    if len(valid_indices) > max_nodes:
        np.random.seed(args.seed)
        valid_indices = np.random.choice(valid_indices, size=max_nodes, replace=False).tolist()
       
    print(f"Nodes to Explain: {len(valid_indices)}")
   
    runner = ExplanationRunner(config['explainer'])
   
    exp_full_path = os.path.join(output_dir, 'explanations', 'explanations_full.pkl')
    if os.path.exists(exp_full_path):
        print(f"Loading existing exp_full from {exp_full_path}...")
        with open(exp_full_path, 'rb') as f:
            exp_full = importlib.import_module('pickle').load(f) if 'importlib' in globals() else __import__('pickle').load(f)
    else:
        print("Explaining GNNFull...")
        exp_full = runner.run(model_full, data, valid_indices)
        with open(exp_full_path, 'wb') as f:
            pickle.dump(exp_full, f)
       
    num_seeds = config['experiment'].get('num_seeds', 1)
   
    all_exp_sampled = []
    trackers = []
    accs_sampled = []
   
    for seed_idx in range(num_seeds):
        curr_seed = args.seed + seed_idx
        print(f"--- GNNSampled Seed {seed_idx + 1}/{num_seeds} ---")
        set_seed(curr_seed)
        
        sampled_model_path = os.path.join(output_dir, 'models', f'gnn_sampled_seed_{seed_idx}.pt')
        sampled_exp_path = os.path.join(output_dir, 'explanations', f'exp_sampled_seed_{seed_idx}.pkl')
        tracker_path = os.path.join(output_dir, 'explanations', f'tracker_seed_{seed_idx}.pkl')
        
        if os.path.exists(sampled_model_path) and os.path.exists(sampled_exp_path) and os.path.exists(tracker_path):
            print(f"Loading existing GNNSampled Seed {seed_idx + 1} from disk...")
            model_sampled = GNNSampled(
                in_channels, config['model']['hidden_dim'], out_channels, 
                num_layers=config['model']['num_layers'], dropout=config['model']['dropout'], seed=curr_seed
            ).to(device)
            model_sampled.load_state_dict(torch.load(sampled_model_path))
            
            with open(sampled_exp_path, 'rb') as f:
                exp_sampled_seed = pickle.load(f)
            with open(tracker_path, 'rb') as f:
                tracker = pickle.load(f)
                
            acc_sampled = evaluate(model_sampled, data, data.test_mask)
        else:
            model_sampled = GNNSampled(
                in_channels, config['model']['hidden_dim'], out_channels, 
                num_layers=config['model']['num_layers'], dropout=config['model']['dropout'], seed=curr_seed
            ).to(device)
            optimizer_sampled = torch.optim.Adam(model_sampled.parameters(), lr=config['model']['lr'])
            tracker = ExclusionTracker(num_nodes=data.num_nodes)
            
            model_sampled = train_sampled(
                model_sampled, data, optimizer_sampled, 
                fanout=config['sampling']['fanout'], batch_size=config['sampling']['batch_size'],
                epochs=config['model']['epochs'], exclusion_tracker=tracker
            )
            torch.save(model_sampled.state_dict(), sampled_model_path)
            
            acc_sampled = evaluate(model_sampled, data, data.test_mask)
            exp_sampled_seed = runner.run(model_sampled, data, valid_indices, desc=f"Explaining Sampled (Seed {seed_idx+1})")
            
            with open(sampled_exp_path, 'wb') as f:
                pickle.dump(exp_sampled_seed, f)
            with open(tracker_path, 'wb') as f:
                pickle.dump(tracker, f)
                
        accs_sampled.append(acc_sampled)
        trackers.append(tracker)
        all_exp_sampled.append(exp_sampled_seed)
       
    print(f"Accuracy - Full: {acc_full:.4f}, Sampled (Avg): {np.mean(accs_sampled):.4f}")
   
    nf_path = os.path.join(output_dir, 'metrics', 'noise_floor.csv')
    if os.path.exists(nf_path):
        print(f"Loading existing Noise Floor from {nf_path}...")
        nf_df = pd.read_csv(nf_path)
        same_model_jaccard = dict(zip(nf_df['node_id'], nf_df['jaccard_same_model']))
    else:
        print("Computing Noise Floor...")
        nf_estimator = NoiseFloorEstimator(runner, n_runs=config['metrics']['noise_floor_runs'])
        same_model_jaccard = nf_estimator.estimate(model_full, data, valid_indices)
        nf_df = pd.DataFrame(list(same_model_jaccard.items()), columns=['node_id', 'jaccard_same_model'])
        nf_df.to_csv(nf_path, index=False)
   
    print("Computing Agreement Metrics across seeds...")
    final_cross_jaccard = {idx: 0.0 for idx in valid_indices}
   
    for exp_sampled_seed in all_exp_sampled:
        agreement = compute_agreement(exp_full, exp_sampled_seed)
        for idx in valid_indices:
            final_cross_jaccard[idx] += agreement.per_node_jaccard.get(idx, 0.0)
           
    for idx in valid_indices:
        final_cross_jaccard[idx] /= num_seeds
       
    print("Computing SOFIA-Index...")
    df_metas = []
    for tracker in trackers:
        df_meta = tracker.get_dataframe(exp_full, bundle.degree_per_node, bundle.homophily_per_node)
        df_metas.append(df_meta)
       
    df_meta_combined = pd.concat(df_metas).groupby('node_id').mean().reset_index()
   
    index_calculator = InstabilityIndex()
    df_sofia = index_calculator.compute(df_meta_combined, final_cross_jaccard, same_model_jaccard)
    df_sofia.to_csv(os.path.join(output_dir, 'metrics', 'sofia_index.csv'), index=False)
   
    print("Running Regression Analysis...")
    
    # Diagnostics for Exclusion Fraction
    excl_scores = df_sofia['influential_exclusion_score']
    print(f"\n--- Diagnostics ---")
    print(f"Exclusion Fraction Mean: {excl_scores.mean():.4f}")
    print(f"Exclusion Fraction Std:  {excl_scores.std():.4f}")
    print(f"Exclusion Fraction Min:  {excl_scores.min():.4f}")
    print(f"Exclusion Fraction Max:  {excl_scores.max():.4f}")
    print(f"-------------------\n")

    analysis_corr = ExclusionCorrelationAnalysis()
    results_corr = analysis_corr.analyze(df_sofia)
   
    with open(os.path.join(output_dir, 'metrics', 'regression_results.txt'), 'w') as f:
        for k, v in results_corr.items():
            f.write(f"{k}: {v}\n")
           
    analysis_hom = HomophilyStratifiedAnalysis()
    results_hom = analysis_hom.analyze(df_sofia)
    results_hom.to_csv(os.path.join(output_dir, 'analysis', 'homophily_stratified.csv'), index=False)
   
    analysis_deg = DegreeControlAnalysis()
    results_deg = analysis_deg.analyze(df_sofia)
    results_deg.to_csv(os.path.join(output_dir, 'analysis', 'degree_stratified.csv'), index=False)
   
    print("Generating Figures...")
    nodes_to_plot = valid_indices[:3]
    plot_explanation_heatmaps(exp_full, all_exp_sampled[0], nodes_to_plot, os.path.join(output_dir, 'figures'), args.dataset)
    plot_agreement_distributions(df_sofia, os.path.join(output_dir, 'figures'), args.dataset)
    plot_instability_by_degree(df_sofia, os.path.join(output_dir, 'figures'), args.dataset)
   
    mean_homophily = df_meta_combined['homophily'][df_meta_combined['homophily'] >= 0].mean()
    mean_cross_jaccard = df_sofia['jaccard_cross'].mean()
    mean_same_jaccard = df_sofia['jaccard_same'].mean()
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
    print(f"p-value (Multi):    {results_corr['multivariate_p_value_exclusion']:.4e}")
    print(f"Partial R^2 (Excl): {results_corr['multivariate_partial_r2_exclusion']:.4f}")
    print(f"Beta (Binary Top-10 Split):{results_corr['multivariate_binary_coef_exclusion']:.4f}")
    print(f"p-value (Binary):   {results_corr['multivariate_binary_p_value_exclusion']:.4e}")
    print("="*80)

if __name__ == "__main__":
    main()
