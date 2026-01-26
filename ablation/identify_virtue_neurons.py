"""
Identifies virtue-only neurons for ablation experiments - as v-only were not identified in the main analysis.
Groups:
1. All virtue-related neurons (V, VD, VU, VDU) - Cohen's d > 0.8
2. Virtue-only neurons (V only)
3. Top virtue-only neurons (10% highest Cohen's d)
"""
import pandas as pd
import numpy as np
import config
import os

def identify_virtue_neurons():
    # Load specialization results
    spec_path = os.path.join(config.RESULTS_DIR, 'master_specialisation_results.csv')
    df = pd.read_csv(spec_path)
    
    # Apply FDR correction and effect size threshold
    p_threshold = 0.05
    d_threshold = 0.8
    
    virtue_specialized = df[(df['v_p_value_fdr'] < p_threshold) & (df['v_cohens_d'] > d_threshold)]
    deont_specialized = df[(df['d_p_value_fdr'] < p_threshold) & (df['d_cohens_d'] > d_threshold)]
    util_specialized = df[(df['u_p_value_fdr'] < p_threshold) & (df['u_cohens_d'] > d_threshold)]
    
    # Create sets of neuron identifiers (layer, index) - MATCHING ORIGINAL ANALYSIS
    virtue_set = set(virtue_specialized.apply(lambda row: (row['layer'], row['neuron_index']), axis=1))
    deont_set = set(deont_specialized.apply(lambda row: (row['layer'], row['neuron_index']), axis=1))
    util_set = set(util_specialized.apply(lambda row: (row['layer'], row['neuron_index']), axis=1))
    
    # Group 1: All virtue-related neurons (V, VD, VU, VDU)
    all_virtue_indices = virtue_set
    all_virtue_df = virtue_specialized.copy()
    all_virtue_df['group'] = 'all_virtue'
    
    # Group 2: Virtue-only neurons (using set operations like original analysis)
    virtue_only_indices = virtue_set - deont_set - util_set
    virtue_only_df = virtue_specialized[
        virtue_specialized.apply(lambda row: (row['layer'], row['neuron_index']) in virtue_only_indices, axis=1)
    ].copy()
    virtue_only_df['group'] = 'virtue_only'
    
    # Group 3: Top virtue-only neurons (top 10% by Cohen's d)
    if len(virtue_only_df) > 0:
        threshold = virtue_only_df['v_cohens_d'].quantile(0.9)  # Top 10%
        top_virtue_df = virtue_only_df[virtue_only_df['v_cohens_d'] >= threshold].copy()
        top_virtue_df['group'] = 'top_virtue'
    else:
        top_virtue_df = pd.DataFrame(columns=virtue_only_df.columns)
    
    # Combine all groups
    all_groups = pd.concat([all_virtue_df, virtue_only_df, top_virtue_df], ignore_index=True)
    
    # Keep only essential columns
    all_groups = all_groups[['layer', 'neuron_index', 'v_cohens_d', 'v_mean_diff', 'group']]
    
    # Save results
    output_path = os.path.join(config.STATS_DIR, 'virtue_ablation_neurons.csv')
    all_groups.to_csv(output_path, index=False)
    
    # Print summary
    print("Virtue Neuron Groups Summary:")
    print(f"All virtue-related neurons: {len(all_virtue_df)}")
    print(f"Virtue-only neurons: {len(virtue_only_df)}")
    print(f"Top virtue-only neurons: {len(top_virtue_df)}")
    
    if len(virtue_only_df) > 0:
        print(f"Top virtue threshold (Cohen's d): {threshold:.3f}")
    
    return all_groups

if __name__ == "__main__":
    identify_virtue_neurons()