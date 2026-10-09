import pandas as pd
class InstabilityIndex:
    def __init__(self, eps=1e-6):
        self.eps = eps
        
    def compute(self, df_metadata, cross_model_jaccard, same_model_jaccard) -> pd.DataFrame:
        records = []
        for _, row in df_metadata.iterrows():
            v = int(row['node_id'])
            j_cross = cross_model_jaccard.get(v, None)
            j_same = same_model_jaccard.get(v, None)
            
            if j_cross is not None and j_same is not None:
                if j_same < 0.10:
                    continue  # Filter out pure noise baselines to prevent division by zero
                    
                sofia_index = (j_same - j_cross) / (j_same + self.eps)
                
                row_dict = row.to_dict()
                row_dict['jaccard_cross'] = j_cross
                row_dict['jaccard_same'] = j_same
                row_dict['sofia_index'] = sofia_index
                records.append(row_dict)
                
        return pd.DataFrame(records)