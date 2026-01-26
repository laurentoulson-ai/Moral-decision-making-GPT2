"""
Processes moral reasoning text data to extract and save neural network activations 
for different ethical frameworks (virtue ethics, deontology, and utilitarianism). It loads a 
pre-trained model, reads text pairs (moral and neutral) for each framework, computes activations 
for all layers of the model, and saves the paired activations as NumPy arrays for further analysis.
"""
import pandas as pd
import numpy as np
import os
from activation_utils import load_model, extract_activations_all_layers, save_activations
import config

def main():
    model = load_model()
    n_layers = model.cfg.n_layers
    n_neurons = model.cfg.d_mlp
    
    framework_paths = {
        'virtue': config.VIRTUE_PAIRS_PATH,
        'deont': config.DEONT_PAIRS_PATH,
        'util': config.UTIL_PAIRS_PATH
    }
    
    for framework, path in framework_paths.items():
        print(f"Processing {framework} pairs...")
        data = pd.read_csv(path)
        moral_texts = data['moral'].tolist()
        neutral_texts = data['neutral'].tolist()
        n_pairs = len(moral_texts)
        
        print(f"Extracting activations for {framework} moral texts...")
        moral_activations = extract_activations_all_layers(model, moral_texts)
        print(f"Extracting activations for {framework} neutral texts...")
        neutral_activations = extract_activations_all_layers(model, neutral_texts)
        
        paired_activations = np.stack([neutral_activations, moral_activations], axis=1)
        
        save_path = os.path.join(config.ACTIVATION_DIR, f'{framework}_paired_activations.npy')
        save_activations(paired_activations, save_path)
        print(f"Saved {framework} activations of shape {paired_activations.shape} to {save_path}")

if __name__ == "__main__":
    main()