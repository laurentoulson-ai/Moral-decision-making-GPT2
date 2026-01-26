"""
Statistical analysis of ablation results.
Performs t-tests to compare performance degradation.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os
import config

def analyze_ablation_results():
    # Load results
    results_path = os.path.join(config.STATS_DIR, 'ablation_evaluation_results.csv')
    results_df = pd.read_csv(results_path)
    
    # Calculate target logit (logit for correct answer)
    results_df['target_logit'] = np.where(
        results_df['target_answer'] == 'Yes',
        results_df['yes_logit'],
        results_df['no_logit']
    )
    
    # Get baseline target logits for comparison
    baseline_data = results_df[results_df['group'] == 'baseline']
    baseline_logits = baseline_data.set_index(['pair_id', 'context_type'])['target_logit']
    
    # Calculate logit differences
    comparison_results = []
    
    for group in results_df['group'].unique():
        if group == 'baseline':
            continue
            
        group_data = results_df[results_df['group'] == group]
        
        # Merge with baseline to calculate differences
        merged = group_data.merge(
            baseline_data[['pair_id', 'context_type', 'target_logit']],
            on=['pair_id', 'context_type'],
            suffixes=('', '_baseline')
        )
        
        merged['logit_diff'] = merged['target_logit'] - merged['target_logit_baseline']
        merged['accuracy_diff'] = merged['is_correct'] - merged['is_correct_baseline']
        
        # Split by context type
        moral_diffs = merged[merged['context_type'] == 'moral']['logit_diff']
        neutral_diffs = merged[merged['context_type'] == 'neutral']['logit_diff']
        
        # Statistical tests
        moral_t_stat, moral_p_value = stats.ttest_1samp(moral_diffs, 0)
        neutral_t_stat, neutral_p_value = stats.ttest_1samp(neutral_diffs, 0)
        
        # Compare moral vs neutral degradation
        group_comparison_t, group_comparison_p = stats.ttest_ind(
            moral_diffs, neutral_diffs, equal_var=False
        )
        
        comparison_results.append({
            'group': group,
            'moral_mean_logit_diff': moral_diffs.mean(),
            'moral_std_logit_diff': moral_diffs.std(),
            'moral_t_statistic': moral_t_stat,
            'moral_p_value': moral_p_value,
            'neutral_mean_logit_diff': neutral_diffs.mean(),
            'neutral_std_logit_diff': neutral_diffs.std(),
            'neutral_t_statistic': neutral_t_stat,
            'neutral_p_value': neutral_p_value,
            'moral_vs_neutral_t': group_comparison_t,
            'moral_vs_neutral_p': group_comparison_p,
            'n_moral': len(moral_diffs),
            'n_neutral': len(neutral_diffs)
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save statistical analysis
    stats_path = os.path.join(config.STATS_DIR, 'ablation_statistical_analysis.csv')
    comparison_df.to_csv(stats_path, index=False)
    
    print("Statistical Analysis Results:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

if __name__ == "__main__":
    analyze_ablation_results()