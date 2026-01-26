import pandas as pd
import numpy as np
import os
import config

def analyze_layer_specialization():
    # Load the master results
    results_path = os.path.join(config.STATS_DIR, 'master_specialisation_results.csv')
    df = pd.read_csv(results_path)
    
    # Define thresholds
    cohen_threshold = 0.8  # For specialized neurons
    activation_threshold = 0.06  # For strong activation
    
    # Initialize results for each framework
    frameworks = ['v', 'd', 'u']
    framework_names = {'v': 'Virtue Ethics', 'd': 'Deontology', 'u': 'Utilitarianism'}
    
    # Create a comprehensive summary table
    summary_data = []
    
    for framework in frameworks:
        # Create a DataFrame for this framework's layer analysis
        framework_df = pd.DataFrame(index=range(12), columns=[
            'Layer', 
            'Max_Cohens_d', 
            'Count_Cohens_d_above_1.25', 
            'Count_Cohens_d_above_0.8', 
            'Max_Activation_Strength',
            'Count_Activation_above_0.06',
            'Percent_Specialized_Above_0.06'
        ])
        
        framework_df['Layer'] = range(12)
        
        for layer in range(12):
            # Filter data for this layer
            layer_data = df[df['layer'] == layer]
            
            # Get framework-specific columns
            cohen_col = f'{framework}_cohens_d'
            mean_diff_col = f'{framework}_mean_diff'
            
            # Calculate metrics
            max_cohen = layer_data[cohen_col].max()
            count_cohen_1_25 = len(layer_data[layer_data[cohen_col] >= 1.25])
            count_cohen_0_8 = len(layer_data[layer_data[cohen_col] >= cohen_threshold])
            max_activation = layer_data[mean_diff_col].max()
            
            # Count neurons with activation above threshold
            count_activation = len(layer_data[layer_data[mean_diff_col] >= activation_threshold])
            
            # Calculate percentage of specialized neurons with activation above threshold
            specialized_neurons = layer_data[layer_data[cohen_col] >= cohen_threshold]
            if len(specialized_neurons) > 0:
                percent_above = len(specialized_neurons[specialized_neurons[mean_diff_col] >= activation_threshold]) / len(specialized_neurons) * 100
            else:
                percent_above = 0
            
            # Store results
            framework_df.loc[layer, 'Max_Cohens_d'] = max_cohen
            framework_df.loc[layer, 'Count_Cohens_d_above_1.25'] = count_cohen_1_25
            framework_df.loc[layer, 'Count_Cohens_d_above_0.8'] = count_cohen_0_8
            framework_df.loc[layer, 'Max_Activation_Strength'] = max_activation
            framework_df.loc[layer, 'Count_Activation_above_0.06'] = count_activation
            framework_df.loc[layer, 'Percent_Specialized_Above_0.06'] = percent_above
            
            # Add to summary data
            summary_data.append({
                'Framework': framework_names[framework],
                'Layer': layer,
                'Max_Cohens_d': max_cohen,
                'Count_Cohens_d_above_1.25': count_cohen_1_25,
                'Count_Cohens_d_above_0.8': count_cohen_0_8,
                'Max_Activation_Strength': max_activation,
                'Count_Activation_above_0.06': count_activation,
                'Percent_Specialized_Above_0.06': percent_above
            })
        
        # Save framework-specific results
        output_path = os.path.join(config.STATS_DIR, f'{framework}_layer_analysis.csv')
        framework_df.to_csv(output_path, index=False)
        print(f"Saved {framework_names[framework]} analysis to {output_path}")
    
    # Create and save comprehensive summary
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(config.STATS_DIR, 'all_frameworks_layer_analysis.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved comprehensive summary to {summary_path}")
    
    # Print key insights
    print("\n=== KEY INSIGHTS ===")
    
    for framework in frameworks:
        framework_data = summary_df[summary_df['Framework'] == framework_names[framework]]
        
        max_cohen_layer = framework_data.loc[framework_data['Max_Cohens_d'].idxmax(), 'Layer']
        max_cohen_value = framework_data['Max_Cohens_d'].max()
        
        max_activation_layer = framework_data.loc[framework_data['Max_Activation_Strength'].idxmax(), 'Layer']
        max_activation_value = framework_data['Max_Activation_Strength'].max()
        
        max_specialized_layer = framework_data.loc[framework_data['Count_Cohens_d_above_0.8'].idxmax(), 'Layer']
        max_specialized_count = framework_data['Count_Cohens_d_above_0.8'].max()
        
        print(f"\n{framework_names[framework]}:")
        print(f"  - Highest Cohen's d ({max_cohen_value:.2f}) in layer {max_cohen_layer}")
        print(f"  - Highest activation ({max_activation_value:.3f}) in layer {max_activation_layer}")
        print(f"  - Most specialized neurons ({max_specialized_count}) in layer {max_specialized_layer}")
    
    return summary_df

if __name__ == "__main__":
    analyze_layer_specialization()