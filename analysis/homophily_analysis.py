import pandas as pd
import numpy as np

class HomophilyStratifiedAnalysis:
    def __init__(self, n_bins=5):
        self.n_bins = n_bins
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Bin nodes into 5 homophily strata and compute mean SOFIA-Index per stratum.
        """
        # Filter isolated nodes (-1.0) and missing data
        clean_df = df[(df['homophily'] >= 0) & (df['homophily'] <= 1.0)].copy()
        clean_df = clean_df.dropna(subset=['sofia_index'])
        
        bins = np.linspace(0, 1, self.n_bins + 1)
        labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(self.n_bins)]
        
        # Include right edge, but we want exactly 0 to be in the first bin, so we use include_lowest=True
        clean_df['homophily_stratum'] = pd.cut(clean_df['homophily'], bins=bins, labels=labels, include_lowest=True)
        
        summary = clean_df.groupby('homophily_stratum', observed=False).agg(
            mean_sofia_index=('sofia_index', 'mean'),
            std_sofia_index=('sofia_index', 'std'),
            n_nodes=('node_id', 'count')
        ).reset_index()
        
        return summary
