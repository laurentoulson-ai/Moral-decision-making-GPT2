"""
This script analyzes neural activations to identify neurons that are specialized for moral reasoning.
Useage: Run after `get_activations.py` to perform statistical tests on the extracted activations.
The script performs the following steps:
1. Loads precomputed neural activation data for "moral" and "neutral" conditions.
2. Iterates through each layer and neuron in the activation data.
3. For each neuron, performs a statistical t-test to determine if there is a significant difference 
    between the activations in the "moral" and "neutral" conditions.
4. Computes the effect size (Cohen's d) to measure the magnitude of the difference.
5. Filters results to include only neurons where:
    - The p-value from the t-test is below a predefined significance threshold (ALPHA).
    - The effect size exceeds a predefined threshold (EFFECT_THRESH).
    - The mean activation for the "moral" condition is greater than the "neutral" condition.
6. Saves the results of significant neurons, including their layer, neuron index, p-value, effect size, 
    and mean activations, to a CSV file.
"""
import numpy as np
import os
import config
from scipy.stats import ttest_ind
import pandas as pd

# Thresholds
ALPHA = 0.1
EFFECT_THRESH = 0.8

def cohens_d(x, y):
    """Compute Cohen's d for two samples."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0

def main():
    # Load activations
    moral = np.load(os.path.join(config.ACTIVATION_DIR, "moral_activations.npy"))
    neutral = np.load(os.path.join(config.ACTIVATION_DIR, "neutral_activations.npy"))

    n_layers = moral.shape[1]
    n_neurons = moral.shape[2]

    significant_results = []

    for layer in range(n_layers):
        for neuron in range(n_neurons):
            moral_vals = moral[:, layer, neuron]
            neutral_vals = neutral[:, layer, neuron]

            # t-test
            t_stat, p_val = ttest_ind(moral_vals, neutral_vals, equal_var=False)

            # effect size (directional: moral > neutral matters most)
            d = cohens_d(moral_vals, neutral_vals)

            if p_val < ALPHA and abs(d) > EFFECT_THRESH:
                # Keep only if effect favors moral activations
                if np.mean(moral_vals) > np.mean(neutral_vals):
                    significant_results.append({
                        "layer": layer,
                        "neuron": neuron,
                        "p_value": p_val,
                        "effect_size_d": d,
                        "moral_mean": np.mean(moral_vals),
                        "neutral_mean": np.mean(neutral_vals)
                    })

    # Save results
    df = pd.DataFrame(significant_results)
    df.to_csv(os.path.join(config.STATS_DIR, "moral_specialisation_results.csv"), index=False)
    print(f"Saved results for {len(df)} neurons specialised for moral detection.")

if __name__ == "__main__":
    main()

