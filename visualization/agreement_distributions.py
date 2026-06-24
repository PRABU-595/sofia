import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_agreement_distributions(df: pd.DataFrame, output_dir: str, dataset_name: str):
    \"\"\"
    Plots KDE plots of per-node Jaccard for cross-model vs same-model (noise floor).
    \"\"\"
    if df.empty or 'jaccard_cross' not in df.columns or 'jaccard_same' not in df.columns:
        return
        
    plt.figure(figsize=(8, 5))
    
    sns.kdeplot(df['jaccard_same'].dropna(), fill=True, label="Same-Model (Noise Floor)", color="green", alpha=0.5)
    sns.kdeplot(df['jaccard_cross'].dropna(), fill=True, label="Cross-Model (Full vs Sampled)", color="red", alpha=0.5)
    
    plt.title(f"Explanation Agreement Distributions ({dataset_name})")
    plt.xlabel("Jaccard Agreement")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_agreement_distributions.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_agreement_distributions.pdf"))
    plt.close()
