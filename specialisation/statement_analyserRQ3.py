import pandas as pd
import numpy as np
import os
from config import DATA_DIR, ACTIVATION_DIR, STATS_DIR

def get_next_filename(base_name, extension="csv"):
    """Find the next available filename to avoid overwriting"""
    i = 1
    while os.path.exists(os.path.join(STATS_DIR, f"{base_name}_{i}.{extension}")):
        i += 1
    return os.path.join(STATS_DIR, f"{base_name}_{i}.{extension}")

def analyze_neurons_for_framework(framework, neurons_to_analyze):
    """Analyze specific neurons for a given framework"""
    print(f"Processing {framework} framework...")
    
    # Load the paired activations
    activations_path = os.path.join(ACTIVATION_DIR, f'{framework}_paired_activations.npy')
    activations = np.load(activations_path)
    
    # Load the text data
    text_path = os.path.join(DATA_DIR, f'{framework}_500.csv')
    text_data = pd.read_csv(text_path)
    
    # Prepare results
    results = []
    
    for idx, row in text_data.iterrows():
        result_row = {
            'pair_id': idx,
            'moral_text': row['moral'],
            'neutral_text': row['neutral'],
        }
        
        # For each neuron, extract activations and calculate difference
        for neuron_info in neurons_to_analyze:
            layer = neuron_info['layer']
            neuron_index = neuron_info['neuron_index']
            neuron_type = neuron_info.get('type', '')  # 'high_precision' or 'high_output'
            cohens_d = neuron_info.get('cohens_d', None)  # Overall Cohen's d from population analysis
            
            neuron_id = f"{neuron_index}_{layer}"
            if neuron_type:
                neuron_id = f"{neuron_id}_{neuron_type}"
            
            # Get activations for this neuron and statement pair
            neutral_act = activations[idx, 0, layer, neuron_index]
            moral_act = activations[idx, 1, layer, neuron_index]
            act_diff = moral_act - neutral_act
            
            # Add to results
            result_row[f'{neuron_id}_neutral_act'] = neutral_act
            result_row[f'{neuron_id}_moral_act'] = moral_act
            result_row[f'{neuron_id}_difference'] = act_diff
            
            # Store neuron metadata (will be same for all rows, but we'll add it once)
            if idx == 0:  # Only add metadata once
                result_row[f'{neuron_id}_cohens_d'] = cohens_d
                result_row[f'{neuron_id}_type'] = neuron_type
        
        results.append(result_row)
    
    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    output_path = get_next_filename(f'{framework}_statement_analysis')
    results_df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")
    
    return results_df
"""
layer,neuron_index,u_cohens_d,u_mean_diff,type
10,2265,1.3953612450995867,0.3332518578781746,high_output
9,625,1.4406709538529978,0.2914737527094548,high_output
"""
def main():
    # Define which neurons to analyze for each framework
    neurons_to_analyze = {
        'virtue': [
            # High-precision detectors (specialized neuron with high Cohen's d)
            {'layer': 6, 'neuron_index': 1426, 'type': 'high_precision', 'cohens_d': 3.00},
            {'layer': 10, 'neuron_index': 625, 'type': 'high_precision', 'cohens_d': 2.97},
            # High-output activator (loud neuron with high activation difference)
            {'layer': 9, 'neuron_index': 582, 'type': 'high_output', 'cohens_d': 1.60},
            {'layer': 11, 'neuron_index': 764, 'type': 'high_output', 'cohens_d': 1.43},
        ],
        'deont': [
            # High-precision detectors (specialized neuron with high Cohen's d)
            {'layer': 2, 'neuron_index': 475, 'type': 'high_precision', 'cohens_d': 2.10},
            {'layer': 9, 'neuron_index': 2555, 'type': 'high_precision', 'cohens_d': 1.99},
            # High-output activator (loud neuron with high activation difference)
            {'layer': 11, 'neuron_index': 896, 'type': 'high_output', 'cohens_d': 1.02},
            {'layer': 11, 'neuron_index': 974, 'type': 'high_output', 'cohens_d': 1.61},
        ],
        'util': [
            # High-precision detectors (specialized neuron with high Cohen's d)
            {'layer': 2, 'neuron_index': 925, 'type': 'high_precision', 'cohens_d': 4.85},
            {'layer': 0, 'neuron_index': 676, 'type': 'high_precision', 'cohens_d': 4.59},
            # High-output activator (loud neuron with high activation difference)
            {'layer': 10, 'neuron_index': 2265, 'type': 'high_output', 'cohens_d': 1.40},
            {'layer': 9, 'neuron_index': 625, 'type': 'high_output', 'cohens_d': 1.44},
        ]
    }

    # Analyze neurons for each framework
    for framework, neurons in neurons_to_analyze.items():
        analyze_neurons_for_framework(framework, neurons)

if __name__ == "__main__":
    main()