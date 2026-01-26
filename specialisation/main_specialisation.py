"""
Performs statistical analysis on neural activation data to investigate the specialization 
of neurons across different moral reasoning frameworks: virtue ethics, deontology, and utilitarianism. 
The analysis includes:
1. Loading precomputed neural activation data for each framework.
2. Performing within-framework analysis using paired t-tests to assess the significance 
    of activation differentials for each neuron and computing effect sizes (Cohen's d).
3. Conducting between-framework analysis using ANOVA to compare activation differentials 
    across frameworks for each neuron.
4. Applying False Discovery Rate (FDR) correction to adjust p-values for multiple comparisons.
5. Saving the results, including raw and FDR-corrected p-values, effect sizes, and mean 
    differentials, to a CSV file for further analysis.
The script is designed to process activations from a model with 12 layers and 3072 neurons 
per layer, and it assumes the data is stored in a specific directory structure defined in 
the `config` module.
"""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import os
import config

def load_framework_activations(framework):
    path = os.path.join(config.ACTIVATION_DIR, f'{framework}_paired_activations.npy')
    return np.load(path)

def main():
    frameworks = ['virtue', 'deont', 'util']
    activations = {framework: load_framework_activations(framework) for framework in frameworks}
    
    n_layers = 12
    n_neurons = 3072
    n_pairs = activations['virtue'].shape[0]
    
    results = []
    
    for layer in range(n_layers):
        for neuron in range(n_neurons):
            neuron_data = {}
            for framework in frameworks:
                act = activations[framework]
                differentials = act[:, 1, layer, neuron] - act[:, 0, layer, neuron]
                neuron_data[framework] = differentials
            
            # Within-framework analysis: paired t-test for each framework
            within_results = {}
            for framework in frameworks:
                diff = neuron_data[framework]
                t_stat, p_value = stats.ttest_1samp(diff, 0)
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
                within_results[framework] = (p_value, cohens_d, np.mean(diff))
            
            # Between-framework analysis: ANOVA on the mean differentials
            anova_data = [neuron_data[fw] for fw in frameworks]
            f_stat, anova_p = stats.f_oneway(*anova_data)
            
            # Store results for this neuron
            result = {
                'layer': layer,
                'neuron_index': neuron,
                'v_p_value': within_results['virtue'][0],
                'v_cohens_d': within_results['virtue'][1],
                'v_mean_diff': within_results['virtue'][2],
                'd_p_value': within_results['deont'][0],
                'd_cohens_d': within_results['deont'][1],
                'd_mean_diff': within_results['deont'][2],
                'u_p_value': within_results['util'][0],
                'u_cohens_d': within_results['util'][1],
                'u_mean_diff': within_results['util'][2],
                'anova_p_value': anova_p
            }
            results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Apply FDR correction for within-framework p-values
    for framework in ['v', 'd', 'u']:
        p_values = df[f'{framework}_p_value']
        _, fdr_p_values = fdrcorrection(p_values)
        df[f'{framework}_p_value_fdr'] = fdr_p_values
    
    # Apply FDR correction for ANOVA p-values
    _, anova_fdr_p_values = fdrcorrection(df['anova_p_value'])
    df['anova_p_value_fdr'] = anova_fdr_p_values
    
    # Save results
    output_path = os.path.join(config.STATS_DIR, 'master_specialisation_results.csv')
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()