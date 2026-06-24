import pytest
import pandas as pd
from metrics.agreement import compute_jaccard, compute_agreement
from metrics.instability_index import InstabilityIndex

def test_jaccard():
    set1 = {1, 2, 3}
    set2 = {1, 2, 3}
    assert compute_jaccard(set1, set2) == 1.0
    
    set3 = {4, 5, 6}
    assert compute_jaccard(set1, set3) == 0.0
    
    set4 = {3, 4, 5}
    assert compute_jaccard(set1, set4) == 0.2 # 1 / 5

def test_compute_agreement():
    E1 = {
        0: [1, 2, 3],
        1: [4, 5, 6]
    }
    E2 = {
        0: [1, 2, 4], # Jaccard: 2 / 4 = 0.5
        1: [7, 8, 9]  # Jaccard: 0 / 6 = 0.0
    }
    
    report = compute_agreement(E1, E2, k_values=[3])
    assert report.per_node_jaccard[0] == 0.5
    assert report.per_node_jaccard[1] == 0.0
    assert report.mean_jaccard == 0.25

def test_sofia_index():
    df_meta = pd.DataFrame([
        {'node_id': 0, 'degree': 5, 'homophily': 0.5, 'influential_exclusion_score': 0.2},
        {'node_id': 1, 'degree': 3, 'homophily': 0.2, 'influential_exclusion_score': 0.0}
    ])
    
    cross_jaccard = {0: 0.4, 1: 0.8}
    same_jaccard = {0: 0.8, 1: 0.8}
    
    calc = InstabilityIndex(eps=0)
    df_result = calc.compute(df_meta, cross_jaccard, same_jaccard)
    
    # Node 0: (0.8 - 0.4) / 0.8 = 0.5
    assert df_result.loc[0, 'sofia_index'] == 0.5
    
    # Node 1: (0.8 - 0.8) / 0.8 = 0.0
    assert df_result.loc[1, 'sofia_index'] == 0.0
