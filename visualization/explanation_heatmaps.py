import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_explanation_heatmaps(explanations_full, explanations_sampled, nodes_to_plot, output_dir, dataset_name):
    """
    Visualizes side-by-side inclusion (1=included, 0=not) of neighbors in the top-k explanation
    for a sample of nodes.
    """
    if not nodes_to_plot:
        return
        
    n = len(nodes_to_plot)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3*n))
    
    if n == 1:
        axes = [axes]
        
    for i, node_id in enumerate(nodes_to_plot):
        full_neighbors = explanations_full.get(node_id, [])
        samp_neighbors = explanations_sampled.get(node_id, [])
        
        all_neighbors = list(set(full_neighbors).union(set(samp_neighbors)))
        if not all_neighbors:
            continue
            
        # Create inclusion vectors
        full_vec = np.array([1 if x in full_neighbors else 0 for x in all_neighbors]).reshape(1, -1)
        samp_vec = np.array([1 if x in samp_neighbors else 0 for x in all_neighbors]).reshape(1, -1)
        
        # We can highlight disagreements
        diff = np.abs(full_vec - samp_vec)
        
        ax_left = axes[i][0]
        ax_right = axes[i][1]
        
        sns.heatmap(full_vec, cmap="Blues", ax=ax_left, cbar=False, xticklabels=all_neighbors, yticklabels=[f"Node {node_id}"])
        ax_left.set_title("Full Model Top-k")
        
        # Use a custom colormap for the sampled one to highlight differences in red
        # If diff is 1, it's a disagreement. We can annotate the heatmap.
        sns.heatmap(samp_vec, cmap="Blues", ax=ax_right, cbar=False, xticklabels=all_neighbors, yticklabels=[])
        ax_right.set_title("Sampled Model Top-k")
        
        for j, val in enumerate(diff[0]):
            if val == 1:
                # Disagreement! Draw a red box
                rect = plt.Rectangle((j, 0), 1, 1, fill=False, edgecolor='red', lw=3)
                ax_right.add_patch(rect)
                
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_explanation_heatmap_samples.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_explanation_heatmap_samples.pdf"))
    plt.close()
