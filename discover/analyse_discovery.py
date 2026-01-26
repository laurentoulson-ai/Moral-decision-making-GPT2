"""
Processes results from main_discovery.py to identify and evaluate neurons that exhibit moral specialization. 
The script includes the following functionalities:
1. **Data Quality Checks**:
    - Validates the filtered data by checking for implausibly large effect sizes.
    - Flags neurons with unexpected results (e.g., neutral > moral significance).
2. **Visualization**:
    - Generates diagnostic plots to summarize the results:
      - Bar chart showing the count of significant neurons per layer.
      - Histogram of effect size distribution (Cohen's d).
      - Heatmap visualizing effect sizes across layers and neurons.
"""
import pandas as pd
import numpy as np
import os
import config
import matplotlib.pyplot as plt
import seaborn as sns

def data_quality_checks(df):
    """Perform validity checks and return warning flags on the filtered data."""
    warnings = []

    # Check 1: Implausibly large effect sizes
    large_effects = (df['effect_size_d'].abs() > 2).sum()
    if large_effects > 10:
        warnings.append(f"⚠️ {large_effects} neurons have |d|>2 (implausibly large).")
    
    # Check 2: Moral < Neutral flagged as significant (a sanity check)
    reversed_cases = (df['moral_mean'] < df['neutral_mean']).sum()
    if reversed_cases > 0:
        warnings.append(f"⚠️ {reversed_cases} neurons were significant in the opposite direction (neutral > moral), indicating an issue with the discovery script.")

    return warnings

def plot_results(df):
    """Generate and save diagnostic plots from the filtered data."""
    
    # Plot 1: Per-layer count of significant neurons
    plt.figure(figsize=(10,6))
    sig_count = df.groupby("layer").size()
    sig_count.plot(kind="bar", color="skyblue")
    plt.title("Count of Moral-Specialised Neurons per Layer")
    plt.ylabel("Number of significant neurons")
    plt.xlabel("Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "significance_per_layer.png"))
    plt.close()

    # Plot 2: Effect size distribution
    plt.figure(figsize=(10,6))
    sns.histplot(df["effect_size_d"], bins=50, kde=True)
    plt.title("Effect Size Distribution (Cohen's d)")
    plt.xlabel("Effect size d")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "effect_size_distribution.png"))
    plt.close()

    # Plot 3: Heatmap of significant neurons (layer x neuron grid)
    pivot = df.pivot(index="layer", columns="neuron", values="effect_size_d")
    plt.figure(figsize=(14,6))
    sns.heatmap(pivot, cmap="coolwarm", center=0, cbar_kws={'label': 'Effect Size (d)'})
    plt.title("Heatmap of Effect Sizes (Moral vs Neutral)")
    plt.xlabel("Neuron")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOT_DIR, "effect_size_heatmap.png"))
    plt.close()

def main():
    # Load results from main_discovery.py
    try:
        results_path = os.path.join(config.STATS_DIR, "moral_specialisation_results.csv")
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"Error: The results file was not found at {results_path}. Please run main_discovery.py first.")
        return

    # Run quality checks on the loaded data
    warnings = data_quality_checks(df)
    if warnings:
        print("\n".join(warnings))
    else:
        print("✅ No data quality issues detected in the filtered results.")

    # Generate plots
    if not df.empty:
        plot_results(df)
    else:
        print("No significant neurons found to plot.")

    # Summary
    n_sig = len(df)
    print(f"Total specialised neurons analysed: {n_sig}")
    print(f"Plots saved to {config.PLOT_DIR}")

if __name__ == "__main__":
    main()

