from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.stats import spearmanr
import numpy as np

@dataclass
class AgreementReport:
    per_node_jaccard: Dict[int, float]
    per_node_spearman: Dict[int, float]
    per_node_overlap_at_k: Dict[int, Dict[int, float]]
    
    mean_jaccard: float
    mean_spearman: float
    mean_overlap_at_k: Dict[int, float]

def compute_jaccard(set1: set, set2: set) -> float:
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 1.0  # Both empty = identical
    return intersection / union

def compute_spearman(list1: List[int], list2: List[int]) -> float:
    # list1 and list2 are ranked items (highest importance first)
    if not list1 or not list2:
        return 0.0
    
    # We create a union of all items to rank
    all_items = list(set(list1).union(set(list2)))
    
    # If item is missing from a list, assign it a rank at the end
    max_rank = len(all_items) + 1
    
    rank1 = []
    rank2 = []
    
    for item in all_items:
        try:
            r1 = list1.index(item)
        except ValueError:
            r1 = max_rank
            
        try:
            r2 = list2.index(item)
        except ValueError:
            r2 = max_rank
            
        rank1.append(r1)
        rank2.append(r2)
        
    if len(all_items) < 2:
        return 1.0 if rank1 == rank2 else 0.0
        
    rho, _ = spearmanr(rank1, rank2)
    return float(rho) if not np.isnan(rho) else 0.0

def compute_overlap_at_k(list1: List[int], list2: List[int], k_values=[3, 5, 10]) -> Dict[int, float]:
    overlaps = {}
    for k in k_values:
        s1 = set(list1[:k])
        s2 = set(list2[:k])
        
        # Overlap is intersection size divided by min(k, actual_len)
        denom = min(k, max(1, len(s1), len(s2)))
        overlaps[k] = len(s1.intersection(s2)) / denom
    return overlaps

def compute_agreement(E1: Dict[int, List[int]], E2: Dict[int, List[int]], k_values=[3, 5, 10]) -> AgreementReport:
    common_nodes = set(E1.keys()).intersection(set(E2.keys()))
    
    per_node_jaccard = {}
    per_node_spearman = {}
    per_node_overlap = {}
    
    for v in common_nodes:
        l1 = E1[v]
        l2 = E2[v]
        
        per_node_jaccard[v] = compute_jaccard(set(l1), set(l2))
        per_node_spearman[v] = compute_spearman(l1, l2)
        per_node_overlap[v] = compute_overlap_at_k(l1, l2, k_values)
        
    mean_jaccard = float(np.mean(list(per_node_jaccard.values()))) if per_node_jaccard else 0.0
    mean_spearman = float(np.mean(list(per_node_spearman.values()))) if per_node_spearman else 0.0
    
    mean_overlap = {}
    for k in k_values:
        vals = [per_node_overlap[v][k] for v in common_nodes]
        mean_overlap[k] = float(np.mean(vals)) if vals else 0.0
        
    return AgreementReport(
        per_node_jaccard=per_node_jaccard,
        per_node_spearman=per_node_spearman,
        per_node_overlap_at_k=per_node_overlap,
        mean_jaccard=mean_jaccard,
        mean_spearman=mean_spearman,
        mean_overlap_at_k=mean_overlap
    )
