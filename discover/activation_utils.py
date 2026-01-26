"""
Helper functions for model activation extraction and caching
"""
import numpy as np
import torch
from transformer_lens import HookedTransformer
import warnings

warnings.filterwarnings("ignore", message="The current process just got forked")

def load_model():
    """Load GPT-2 model with caching"""
    return HookedTransformer.from_pretrained("gpt2-small")

def extract_activations(model, texts, layer):
    """
    Extract mean-pooled MLP activations for a batch of texts.
    Returns: numpy array of shape [num_texts, n_neurons]
    """
    activations = []
    skipped_count = 0  # Counter for skipped texts

    for text in texts:
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            # Extract activations for the specified layer
            layer_acts = cache[f"blocks.{layer}.mlp.hook_post"]
            
            # Debugging: Print the shape of layer_acts
            print(f"Text: {text[:30]}... Tokens: {tokens.shape}, Layer {layer} activations shape: {layer_acts.shape}")
            
            # Ensure mean pooling over tokens to reduce to [n_neurons]
            if layer_acts.ndim == 3:  # Shape: [batch_size, num_tokens, n_neurons]
                layer_acts_mean = layer_acts.mean(dim=1).squeeze(0).cpu().numpy()  # Shape: [n_neurons]
            else:
                print(f"Skipping text due to unexpected activation shape: {layer_acts.shape}")
                skipped_count += 1
                continue
        
        activations.append(layer_acts_mean)
    
    # Print the number of skipped texts
    print(f"Skipped {skipped_count} texts due to unexpected activation shapes.")
    
    # Ensure all activations are of shape [n_neurons]
    return np.array(activations)

def save_activations(activations, path):
    """Save activations as a numpy file"""
    np.save(path, activations)

def load_activations(path):
    """Load activations from file"""
    return np.load(path)["activations"]