import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
def plot_instability_by_degree(df: pd.DataFrame, output_dir: str, dataset_name: str):
    if df.empty or 'sofia_index' not in df.columns:
        return
        
    clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['degree', 'sofia_index', 'influential_exclusion_score'])
    if clean_df.empty: return
        
    clean_df['log_degree'] = np.log1p(clean_df['degree'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    scatter = ax1.scatter(clean_df['log_degree'], clean_df['sofia_index'], c=clean_df['influential_exclusion_score'], cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter, ax=ax1, label='Influential Neighbor Exclusion Score')
    sns.regplot(x='log_degree', y='sofia_index', data=clean_df, scatter=False, lowess=True, color='red', ax=ax1, line_kws={'label': 'LOWESS'})
    ax1.set_title("Instability vs. Node Degree (Exclusion)")
    ax1.set_xlabel("Log(Degree + 1)")
    ax1.set_ylabel("SOFIA-Index")
    ax1.legend()
    
    scatter2 = ax2.scatter(clean_df['log_degree'], clean_df['sofia_index'], c=clean_df['homophily'], cmap='coolwarm', alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.colorbar(scatter2, ax=ax2, label='Node Homophily')
    sns.regplot(x='log_degree', y='sofia_index', data=clean_df, scatter=False, lowess=True, color='black', ax=ax2, line_kws={'label': 'LOWESS'})
    ax2.set_title("Instability vs. Node Degree (Homophily)")
    ax2.set_xlabel("Log(Degree + 1)")
    ax2.set_ylabel("SOFIA-Index")
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_instability_by_degree.png"), dpi=300)
    plt.close()
    metrics = {
        'degree': 'Node Degree', 'homophily': 'Local Homophily',
        'exclusion_rate': 'Overall Exclusion Rate', 'influential_exclusion_score': 'Influential Exclusion Score',
        'jaccard_same': 'Baseline Agreement', 'jaccard_cross': 'Instability (Cross)', 'sofia_index': 'SOFIA-Index'
    }
    
    for col, title in metrics.items():
        if col not in clean_df.columns: continue
        plt.figure(figsize=(8, 6))
        sns.histplot(clean_df[col].dropna(), bins=50, kde=True, color='blue')
        plt.title(f'Distribution of {title}')
        plt.xlabel(title)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'metric_{col}.png'), dpi=300)
        plt.close()