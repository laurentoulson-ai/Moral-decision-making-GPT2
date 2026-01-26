"""
This script extracts and saves neuron activations from a GPT-2 model for moral and neutral text data.
The program performs the following steps:
1. Loads a pre-trained GPT-2 small model.
2. Reads moral and neutral text datasets from CSV files.
3. Iterates through each layer of the model to extract neuron activations for the given texts.
4. Stacks the activations into numpy arrays for all layers.
5. Saves the stacked activations as `.npy` files for further analysis.
Usage:
Run this script first, then proceed to `main_discovery.py` for statistical analysis of the activations.
"""
import pandas as pd
import numpy as np
import os
import config
from activation_utils import load_model, extract_activations, save_activations

def main():
    # Load GPT-2 model
    print("Loading GPT-2 small model...")
    model = load_model()
    
    # Load moral and neutral text data
    print("Loading text data...")
    moral_data = pd.read_csv(config.MORAL_DATA)
    moral_texts = moral_data['Statement'].tolist()
    
    neutral_data = pd.read_csv(config.NEUTRAL_DATA)
    neutral_texts = neutral_data['Statement'].tolist()
    
    n_layers = model.cfg.n_layers
    n_neurons = model.cfg.d_mlp
    
    moral_activations = []
    neutral_activations = []
    
    # Process each layer
    for layer in range(n_layers):
        print(f"Extracting activations for layer {layer}...")
        
        moral_layer_acts = extract_activations(model, moral_texts, layer)
        neutral_layer_acts = extract_activations(model, neutral_texts, layer)
        
        moral_activations.append(moral_layer_acts)
        neutral_activations.append(neutral_layer_acts)
        
    # Stack activations into a single numpy array
    moral_activations_stacked = np.stack(moral_activations, axis=1)  # Shape: [num_texts, n_layers, n_neurons]
    neutral_activations_stacked = np.stack(neutral_activations, axis=1)
    
    # Save the stacked activations
    print("Saving activations...")
    save_activations(moral_activations_stacked, os.path.join(config.ACTIVATION_DIR, "moral_activations.npy"))
    save_activations(neutral_activations_stacked, os.path.join(config.ACTIVATION_DIR, "neutral_activations.npy"))
    
    print("Activation extraction complete.")
    print(f"Shape of moral activations: {moral_activations_stacked.shape}")
    print(f"Shape of neutral activations: {neutral_activations_stacked.shape}")

if __name__ == "__main__":
    main()