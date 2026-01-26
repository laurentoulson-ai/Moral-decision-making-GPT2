"""
This script analyzes neuron specialization data to identify and select neurons that are exclusively specialized for specific moral reasoning frameworks 
It applies statistical thresholds to determine specialization, identifies exclusive neurons for each framework, and selects top neurons based on precision 
and activation metrics.
Outputs:
- CSV files containing selected neurons for each framework.
- Console logs summarizing the number of exclusive neurons and the range of metrics for selected neurons.
"""
import pandas as pd
import os
import config

# Load the neuron specialization data
results_path = os.path.join(config.STATS_DIR, 'master_specialisation_results.csv')
df = pd.read_csv(results_path)

# Define significance thresholds
alpha = 0.05
d_threshold = 0.8

# Frameworks and their prefixes in the CSV
frameworks = {
    'virtue': 'v',
    'deont': 'd',
    'util': 'u'
}

# Create boolean series for specialized neurons for each framework
is_v_specialized = (df['v_p_value_fdr'] < alpha) & (df['v_cohens_d'] > d_threshold)
is_d_specialized = (df['d_p_value_fdr'] < alpha) & (df['d_cohens_d'] > d_threshold)
is_u_specialized = (df['u_p_value_fdr'] < alpha) & (df['u_cohens_d'] > d_threshold)

# For each framework, identify exclusive neurons (specialized for target, not specialized for others)
exclusive_neurons = {}
exclusive_neurons['virtue'] = df[is_v_specialized & ~is_d_specialized & ~is_u_specialized].copy()
exclusive_neurons['deont'] = df[is_d_specialized & ~is_v_specialized & ~is_u_specialized].copy()
exclusive_neurons['util'] = df[is_u_specialized & ~is_v_specialized & ~is_d_specialized].copy()

print(f"Number of virtue-exclusive neurons: {len(exclusive_neurons['virtue'])}")
print(f"Number of deont-exclusive neurons: {len(exclusive_neurons['deont'])}")
print(f"Number of util-exclusive neurons: {len(exclusive_neurons['util'])}")

# Select top neurons for each framework
selected_neurons = {}
for fw_name, fw_prefix in frameworks.items():
    fw_df = exclusive_neurons[fw_name]
    # High-precision detectors: top 10 by Cohen's d
    high_precision = fw_df.nlargest(10, f'{fw_prefix}_cohens_d')[['layer', 'neuron_index', f'{fw_prefix}_cohens_d', f'{fw_prefix}_mean_diff']]
    high_precision['type'] = 'high_precision'
    # High-output activators: top 10 by mean activation difference (all layers)
    high_output = fw_df.nlargest(10, f'{fw_prefix}_mean_diff')[['layer', 'neuron_index', f'{fw_prefix}_mean_diff', f'{fw_prefix}_cohens_d']]
    high_output['type'] = 'high_output'
    selected_neurons[fw_name] = pd.concat([high_precision, high_output], ignore_index=True)
    
    # Save to CSV
    output_path = os.path.join(config.STATS_DIR, f'{fw_name}_selected_neurons.csv')
    selected_neurons[fw_name].to_csv(output_path, index=False)
    print(f"Saved {fw_name} selected neurons to {output_path}")

    # Print summary
    print(f"\n{fw_name.capitalize()} High-Precision Neurons - Cohen's d range: {high_precision[f'{fw_prefix}_cohens_d'].min():.3f} to {high_precision[f'{fw_prefix}_cohens_d'].max():.3f}")
    print(f"{fw_name.capitalize()} High-Output Neurons - Mean activation difference range: {high_output[f'{fw_prefix}_mean_diff'].min():.3f} to {high_output[f'{fw_prefix}_mean_diff'].max():.3f}")