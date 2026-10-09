import pandas as pd
import numpy as np
from scipy import stats

class DegreeControlAnalysis:
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin nodes by log-degree into quintiles and analyze exclusion vs SOFIA-Index.
        """
        clean_df = df[df['degree'] > 0].copy()
        clean_df = clean_df.dropna(subset=['sofia_index', 'influential_exclusion_score'])
        
        clean_df['log_degree'] = np.log1p(clean_df['degree'])
        
        try:
            clean_df['degree_quintile'] = pd.qcut(clean_df['log_degree'], q=self.n_bins, labels=[f"Q{i+1}" for i in range(self.n_bins)])
        except ValueError:
            # If there are duplicate edges making quantiles impossible
            clean_df['degree_quintile'] = pd.cut(clean_df['log_degree'], bins=self.n_bins, labels=[f"Bin{i+1}" for i in range(self.n_bins)])
            
        results = []
        for quintile, group in clean_df.groupby('degree_quintile', observed=False):
            if len(group) > 2:
                x = group['influential_exclusion_score'].values
                y = group['sofia_index'].values
                slope, _, r_val, p_val, _ = stats.linregress(x, y)
                
                results.append({
                    'degree_stratum': quintile,
                    'mean_degree': group['degree'].mean(),
                    'slope_exclusion': slope,
                    'r_value': r_val,
                    'p_value': p_val,
                    'n_nodes': len(group)
                })
                
        return pd.DataFrame(results)
