# SOFIA Experimental Results Summary: Three-Dataset Comparison
**Manuscript:** *SOFIA: Quantifying Sampling-Induced Explanation Instability in Graph Neural Networks*  
**Submission ID:** `dd5571a4-58cd-45d4-bbf0-17059862ef5e` (Scientific Reports)

| Dataset | Graph Domain | Homophily | Avg Degree | Excl. Rate (%) | Infl. Excl. Score (%) | Noise Floor (Same Jaccard) | Cross Jaccard | Instability Gap | SOFIA-Index | Beta Exclusion | p-value | Partial R² | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Cora** | Citation | 0.771 | 6.14 | 3.1% | 6.6% | 0.718 | 0.653 | 0.064 | 0.089 | -0.116 | 0.401 | 0.006 | **Null Result** (Structural Control) |
| **Actor** | Heterophilous | 0.228 | 6.01 | 39.8% | 71.1% | 0.791 | 0.702 | 0.089 | 0.131 | **0.255** | < 1e-15 | **0.130** | **Significant** (H2 Proven: Signal concentration) |
| **Reddit** | Social Network | 0.816 | 143.46 | 87.2% | 89.7% | 0.538 | 0.292 | 0.246 | 0.440 | **0.687** | 3.33e-15 | **0.095** | **Significant** (H1 Proven: Aggressive volume) |

### Key Experimental Insights Supporting the Findings:
1. **Hypothesis 1 (Sampling-Induced Instability):**
   - High-degree nodes undergo substantial neighborhood exclusion during mini-batch sampling, introducing severe explanation drift (Reddit gap = 0.246, Beta = 0.687, p = 3.33e-15).
   - In contrast, low-degree homophilous graphs (Cora) serve as a natural control: with low degrees, neighbor exclusion is negligible (3.1%), and explanation drift remains within the stochastic noise floor (p = 0.401, Partial R² ~ 0).
2. **Hypothesis 2 (Heterophily & Signal Concentration Mechanism):**
   - In heterophilous graphs (Actor, homophily = 0.228), neighbors are informative and non-redundant. Even at a moderate exclusion rate, dropping critical neighbors induces significant explanation drift (Beta = 0.255, p < 1e-15).
   - Crucially, exclusion fraction explains **13.0% of explanation instability variance** (Partial R² = 0.130) after rigorously controlling for node degree.
3. **Accuracy Parity Controls for Confounders:**
   - Full model accuracy vs. Sampled model accuracy across all datasets exhibits parity (Reddit: 0.944 vs 0.947; Actor: 0.326 vs 0.329), proving that explanation instability is not an artifact of degraded classification accuracy.
