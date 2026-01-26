"""
This script analyzes the specialization of neurons in the network with respect to three moral reasoning frameworks: 
Virtue Ethics, Deontology, and Utilitarianism. It performs the following tasks:
1. Loads neuron specialization data and activation data for each framework.
2. Identifies neurons specialized for each framework based on statistical thresholds.
3. Analyzes the overlap of specialized neurons across the three frameworks and generates a Venn diagram.
4. Compares activation strengths for neurons shared between frameworks and calculates activation ratios.
5. Visualizes activation strength comparisons and distributions using scatter plots, box plots, and heatmaps.
6. Saves analysis results, including overlap statistics, activation comparisons, and visualizations, to specified directories.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn3
from scipy import stats
import os
import config

def load_framework_activations(framework):
    """Load the paired activations for a framework"""
    path = os.path.join(config.ACTIVATION_DIR, f'{framework}_paired_activations.npy')
    return np.load(path)

def main():
    # Load results
    results_path = os.path.join(config.STATS_DIR, 'master_specialisation_results.csv')
    df = pd.read_csv(results_path)
    
    # Load activation data for all frameworks
    virtue_acts = load_framework_activations('virtue')
    deont_acts = load_framework_activations('deont')
    util_acts = load_framework_activations('util')
    
    # Define specialization thresholds
    p_threshold = 0.05
    d_threshold = 0.8
    
    # Identify specialized neurons for each framework
    virtue_specialized = df[(df['v_p_value_fdr'] < p_threshold) & (df['v_cohens_d'] > d_threshold)]
    deont_specialized = df[(df['d_p_value_fdr'] < p_threshold) & (df['d_cohens_d'] > d_threshold)]
    util_specialized = df[(df['u_p_value_fdr'] < p_threshold) & (df['u_cohens_d'] > d_threshold)]
    
    # Create sets of neuron identifiers (layer, index)
    virtue_set = set(virtue_specialized.apply(lambda row: (row['layer'], row['neuron_index']), axis=1))
    deont_set = set(deont_specialized.apply(lambda row: (row['layer'], row['neuron_index']), axis=1))
    util_set = set(util_specialized.apply(lambda row: (row['layer'], row['neuron_index']), axis=1))
    
    # Overlap analysis
    overlap_text = f"""
    Overlap Analysis:
    Total specialized neurons in any framework: {len(virtue_set | deont_set | util_set)}
    Virtue-only: {len(virtue_set - deont_set - util_set)}
    Deont-only: {len(deont_set - virtue_set - util_set)}
    Util-only: {len(util_set - virtue_set - deont_set)}
    Virtue and Deont: {len(virtue_set & deont_set - util_set)}
    Virtue and Util: {len(virtue_set & util_set - deont_set)}
    Deont and Util: {len(deont_set & util_set - virtue_set)}
    All three: {len(virtue_set & deont_set & util_set)}
    """
    
    print(overlap_text)
    
    # Save overlap analysis
    with open(os.path.join(config.STATS_DIR, 'overlap_analysis.txt'), 'w') as f:
        f.write(overlap_text)
    
    # Create Venn diagram
    plt.figure(figsize=(10, 8))
    venn3([virtue_set, deont_set, util_set], ('Virtue', 'Deont', 'Util'))
    plt.title("Overlap of Moral Framework Specialized Neurons")
    plt.savefig(os.path.join(config.PLOT_DIR, 'framework_overlap_venn.png'))
    plt.close()
    
    # Analysis of activation strength differences for shared neurons
    # Get all neurons that are specialized in at least two frameworks
    shared_neurons = []
    for layer, neuron in (virtue_set & deont_set):
        shared_neurons.append(('VD', layer, neuron))
    for layer, neuron in (virtue_set & util_set):
        shared_neurons.append(('VU', layer, neuron))
    for layer, neuron in (deont_set & util_set):
        shared_neurons.append(('DU', layer, neuron))
    for layer, neuron in (virtue_set & deont_set & util_set):
        shared_neurons.append(('VDU', layer, neuron))
    
    # For each shared neuron, extract the mean activation values for moral statements
    activation_comparisons = []
    
    for combo, layer, neuron in shared_neurons:
        # Convert layer and neuron identifiers to integers INSIDE the loop
        layer_int = int(layer)
        neuron_int = int(neuron)
        # Extract mean activation for moral statements for each relevant framework
        v_act = virtue_acts[:, 1, layer_int, neuron_int].mean() if 'V' in combo else np.nan
        d_act = deont_acts[:, 1, layer_int, neuron_int].mean() if 'D' in combo else np.nan
        u_act = util_acts[:, 1, layer_int, neuron_int].mean() if 'U' in combo else np.nan
        
        activation_comparisons.append({
            'combo': combo,
            'layer': layer,
            'neuron': neuron,
            'v_activation': v_act,
            'd_activation': d_act,
            'u_activation': u_act
        })
    
    # Convert to DataFrame
    act_df = pd.DataFrame(activation_comparisons)
    
    # Calculate activation ratios for shared neurons
    for combo in ['VD', 'VU', 'DU', 'VDU']:
        combo_df = act_df[act_df['combo'] == combo]
        if len(combo_df) > 0:
            if combo == 'VD':
                combo_df['v_d_ratio'] = combo_df['v_activation'] / combo_df['d_activation']
                print(f"\nVirtue vs Deont activation ratios (n={len(combo_df)}):")
                print(f"Mean ratio: {combo_df['v_d_ratio'].mean():.3f}")
                print(f"Virtue stronger: {(combo_df['v_d_ratio'] > 1).sum()}")
                print(f"Deont stronger: {(combo_df['v_d_ratio'] < 1).sum()}")
                
            elif combo == 'VU':
                combo_df['v_u_ratio'] = combo_df['v_activation'] / combo_df['u_activation']
                print(f"\nVirtue vs Util activation ratios (n={len(combo_df)}):")
                print(f"Mean ratio: {combo_df['v_u_ratio'].mean():.3f}")
                print(f"Virtue stronger: {(combo_df['v_u_ratio'] > 1).sum()}")
                print(f"Util stronger: {(combo_df['v_u_ratio'] < 1).sum()}")
                
            elif combo == 'DU':
                combo_df['d_u_ratio'] = combo_df['d_activation'] / combo_df['u_activation']
                print(f"\nDeont vs Util activation ratios (n={len(combo_df)}):")
                print(f"Mean ratio: {combo_df['d_u_ratio'].mean():.3f}")
                print(f"Deont stronger: {(combo_df['d_u_ratio'] > 1).sum()}")
                print(f"Util stronger: {(combo_df['d_u_ratio'] < 1).sum()}")
    
    # Visualize activation strength differences for shared neurons
    if len(act_df) > 0:
        # Create a figure with subplots for each combination
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # VD comparison
        vd_df = act_df[act_df['combo'].str.contains('VD')]
        if len(vd_df) > 0:
            axes[0, 0].scatter(vd_df['v_activation'], vd_df['d_activation'], alpha=0.5)
            min_val = min(vd_df['v_activation'].min(), vd_df['d_activation'].min())
            max_val = max(vd_df['v_activation'].max(), vd_df['d_activation'].max())
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[0, 0].set_xlabel('Virtue Mean Activation')
            axes[0, 0].set_ylabel('Deont Mean Activation')
            axes[0, 0].set_title('Virtue vs Deont Activation Strength')

        # VU comparison
        vu_df = act_df[act_df['combo'].str.contains('VU')]
        if len(vu_df) > 0:
            axes[0, 1].scatter(vu_df['v_activation'], vu_df['u_activation'], alpha=0.5)
            min_val = min(vu_df['v_activation'].min(), vu_df['u_activation'].min())
            max_val = max(vu_df['v_activation'].max(), vu_df['u_activation'].max())
            axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[0, 1].set_xlabel('Virtue Mean Activation')
            axes[0, 1].set_ylabel('Util Mean Activation')
            axes[0, 1].set_title('Virtue vs Util Activation Strength')

        # DU comparison
        du_df = act_df[act_df['combo'].str.contains('DU')]
        if len(du_df) > 0:
            axes[1, 0].scatter(du_df['d_activation'], du_df['u_activation'], alpha=0.5)
            min_val = min(du_df['d_activation'].min(), du_df['u_activation'].min())
            max_val = max(du_df['d_activation'].max(), du_df['u_activation'].max())
            axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
            axes[1, 0].set_xlabel('Deont Mean Activation')
            axes[1, 0].set_ylabel('Util Mean Activation')
            axes[1, 0].set_title('Deont vs Util Activation Strength')
        
        # Distribution of activation ratios
        ratios = []
        labels = []
        
        if len(vd_df) > 0:
            ratios.extend(vd_df['v_activation'] / vd_df['d_activation'])
            labels.extend(['V/D'] * len(vd_df))
        
        if len(vu_df) > 0:
            ratios.extend(vu_df['v_activation'] / vu_df['u_activation'])
            labels.extend(['V/U'] * len(vu_df))
        
        if len(du_df) > 0:
            ratios.extend(du_df['d_activation'] / du_df['u_activation'])
            labels.extend(['D/U'] * len(du_df))
        
        if ratios:
            ratio_df = pd.DataFrame({'ratio': ratios, 'comparison': labels})
            sns.boxplot(x='comparison', y='ratio', data=ratio_df, ax=axes[1, 1])
            axes[1, 1].axhline(1, color='r', linestyle='--')
            axes[1, 1].set_ylabel('Activation Ratio')
            axes[1, 1].set_title('Distribution of Activation Ratios')
            axes[1, 1].set_yscale('log')  # Use log scale to better visualize ratios
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOT_DIR, 'activation_strength_comparisons.png'))
        plt.close()
        
        # Create a heatmap of activation strengths by layer for ALL specialized neurons
        all_specialized_activations = []
        # Process Virtue neurons
        for layer, neuron in virtue_set:
            mean_act = virtue_acts[:, 1, int(layer), int(neuron)].mean()
            all_specialized_activations.append({'layer': layer, 'framework': 'v_activation', 'activation': mean_act})
        # Process Deontology neurons
        for layer, neuron in deont_set:
            mean_act = deont_acts[:, 1, int(layer), int(neuron)].mean()
            all_specialized_activations.append({'layer': layer, 'framework': 'd_activation', 'activation': mean_act})
        # Process Utilitarianism neurons
        for layer, neuron in util_set:
            mean_act = util_acts[:, 1, int(layer), int(neuron)].mean()
            all_specialized_activations.append({'layer': layer, 'framework': 'u_activation', 'activation': mean_act})

        # Create DataFrame and pivot for heatmap
        heatmap_df = pd.DataFrame(all_specialized_activations)
        pivot_df = heatmap_df.pivot_table(index='framework', columns='layer', values='activation', aggfunc='mean')

        # Plot the heatmap
        plt.figure(figsize=(12, 5))
        sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
        plt.title('Mean Activation Strength by Layer and Framework')
        plt.ylabel('Framework')
        plt.xlabel('Layer')
        plt.tight_layout()
        plt.savefig(os.path.join(config.PLOT_DIR, 'activation_by_layer_heatmap.png'))
        plt.close()
        
        # Save the activation comparison data for further analysis
        act_df.to_csv(os.path.join(config.STATS_DIR, 'activation_comparisons.csv'), index=False)

if __name__ == "__main__":
    main()