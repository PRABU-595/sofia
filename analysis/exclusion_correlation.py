import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

class ExclusionCorrelationAnalysis:
    def __init__(self):
        pass
       
    def analyze(self, df: pd.DataFrame) -> dict:
        clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[
            'sofia_index', 'influential_exclusion_score', 'degree', 'homophily'
        ])
        y = clean_df['sofia_index'].values
       
        x_univ = clean_df['influential_exclusion_score'].values
        slope_univ, intercept_univ, r_value_univ, p_value_univ, std_err_univ = stats.linregress(x_univ, y)
       
        X_multi = clean_df[['influential_exclusion_score', 'degree', 'homophily']].values
        model = LinearRegression().fit(X_multi, y)
        n = X_multi.shape[0]
        p = X_multi.shape[1] + 1
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
        except np.linalg.LinAlgError:
            p_values = [np.nan] * p
           
        X_reduced = clean_df[['degree', 'homophily']].values
        model_reduced = LinearRegression().fit(X_reduced, y)
        rss_full = np.sum((y - model.predict(X_multi))**2)
        rss_reduced = np.sum((y - model_reduced.predict(X_reduced))**2)
        partial_r2_exclusion = (rss_reduced - rss_full) / rss_reduced if rss_reduced > 0 else 0.0
           
        median_exclusion = clean_df['influential_exclusion_score'].median()
        if median_exclusion == 0.0:
            median_exclusion = 0.5
           
        clean_df['lost_top_k_neighbor'] = (clean_df['influential_exclusion_score'] > median_exclusion).astype(float)
       
        if clean_df['lost_top_k_neighbor'].nunique() < 2:
            model_bin_coef = 0.0
            p_values_bin_1 = np.nan
        else:
            X_multi_binary = clean_df[['lost_top_k_neighbor', 'degree', 'homophily']].values
            model_bin = LinearRegression().fit(X_multi_binary, y)
            X_design_bin = np.column_stack((np.ones(n), X_multi_binary))
            residuals_bin = y - model_bin.predict(X_multi_binary)
           
            try:
                residual_sum_of_squares_bin = np.sum(residuals_bin ** 2)
                sigma_squared_bin = residual_sum_of_squares_bin / (n - p)
                var_b_bin = sigma_squared_bin * np.linalg.inv(np.dot(X_design_bin.T, X_design_bin)).diagonal()
                std_errs_bin = np.sqrt(var_b_bin)
                t_stats_bin = np.append(model_bin.intercept_, model_bin.coef_) / std_errs_bin
                p_values_bin = [2 * (1 - stats.t.cdf(np.abs(t), n - p)) for t in t_stats_bin]
                model_bin_coef = model_bin.coef_[0]
                p_values_bin_1 = p_values_bin[1]
            except np.linalg.LinAlgError:
                model_bin_coef = 0.0
                p_values_bin_1 = np.nan
           
        return {
            'univariate_r': r_value_univ,
            'univariate_p_value': p_value_univ,
            'multivariate_coef_exclusion': model.coef_[0],
            'multivariate_p_value_exclusion': p_values[1],
            'multivariate_partial_r2_exclusion': partial_r2_exclusion,
            'multivariate_binary_coef_exclusion': model_bin_coef,
            'multivariate_binary_p_value_exclusion': p_values_bin_1
        }
