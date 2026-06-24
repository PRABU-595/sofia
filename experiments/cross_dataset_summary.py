import pandas as pd
import os

if __name__ == "__main__":
    # Ensure the summary table is created
    summary_file = 'outputs/results_summary.csv'
    if os.path.exists(summary_file):
        df = pd.read_csv(summary_file, names=[
            'Dataset', 'Homophily', 'Cross-Model Jaccard', 'Same-Model Jaccard (NF)', 
            'SOFIA-Index (mean)', 'beta_exclusion', 'p-value'
        ])
        print(df.to_markdown(index=False))
    else:
        print("No results_summary.csv found. Run experiments first.")
