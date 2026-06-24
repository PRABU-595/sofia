import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

class ExclusionCorrelationAnalysis:
    def __init__(self):
        pass
        
    def analyze(self, df: pd.DataFrame) -> dict:
        \"\"\"
        Regress SOFIA-Index on influential_exclusion_score controlling for degree and homophily.
        df must contain: 'sofia_index', 'influential_exclusion_score', 'degree', 'homophily'
        \"\"\"
        # Drop NaN or inf
        clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
            'sofia_index', 'influential_exclusion_score', 'degree', 'homophily'
        ])
        
        y = clean_df['sofia_index'].values
        
        # Univariate
        x_univ = clean_df['influential_exclusion_score'].values
        slope_univ, intercept_univ, r_value_univ, p_value_univ, std_err_univ = stats.linregress(x_univ, y)
        
        # Multivariate: SOFIA ~ exclusion + degree + homophily
        X_multi = clean_df[['influential_exclusion_score', 'degree', 'homophily']].values
        model = LinearRegression()
        model.fit(X_multi, y)
        
        # We need p-values for multivariate, statsmodels is better but we use sklearn/scipy hybrid approach
        # Let's calculate standard errors for the coefficients manually to get p-values.
        n = X_multi.shape[0]
        p = X_multi.shape[1] + 1 # +1 for intercept
        
        X_design = np.column_stack((np.ones(n), X_multi))
        y_pred = model.predict(X_multi)
        residuals = y - y_pred
        
        try:
            residual_sum_of_squares = np.sum(residuals ** 2)
            sigma_squared = residual_sum_of_squares / (n - p)
            var_b = sigma_squared * np.linalg.inv(np.dot(X_design.T, X_design)).diagonal()
            std_errs = np.sqrt(var_b)
            t_stats = np.append(model.intercept_, model.coef_) / std_errs
            p_values = [2 * (1 - stats.t.cdf(np.abs(t), n - p)) for t in t_stats]
            
            coef_exclusion_multi = model.coef_[0]
            p_value_exclusion_multi = p_values[1] # index 1 is the first feature
            
            # Partial R^2 (incremental R^2 of adding exclusion score)
            # 1. Fit without exclusion score
            X_reduced = clean_df[['degree', 'homophily']].values
            model_reduced = LinearRegression().fit(X_reduced, y)
            r2_full = model.score(X_multi, y)
            r2_reduced = model_reduced.score(X_reduced, y)
            partial_r2 = (r2_full - r2_reduced) / (1 - r2_reduced)
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            coef_exclusion_multi = slope_univ
            p_value_exclusion_multi = p_value_univ
            partial_r2 = r_value_univ ** 2
            
        return {
            'univariate_slope': slope_univ,
            'univariate_p_value': p_value_univ,
            'multivariate_coef_exclusion': coef_exclusion_multi,
            'multivariate_p_value_exclusion': p_value_exclusion_multi,
            'partial_r2': partial_r2,
            'n_samples': n
        }
