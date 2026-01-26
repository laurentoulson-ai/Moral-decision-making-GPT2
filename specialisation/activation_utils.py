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

def extract_activations_all_layers(model, texts):
    """
    Extract activations for all layers for a list of texts.
    Returns: numpy array of shape [num_texts, n_layers, n_neurons]
    """
    n_layers = model.cfg.n_layers
    n_neurons = model.cfg.d_mlp
    num_texts = len(texts)
    activations = np.zeros((num_texts, n_layers, n_neurons))
    
    for i, text in enumerate(texts):
        tokens = model.to_tokens(text)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            for layer in range(n_layers):
                layer_acts = cache[f"blocks.{layer}.mlp.hook_post"]
                if layer_acts.ndim == 3:
                    layer_acts_mean = layer_acts.mean(dim=1).squeeze(0).cpu().numpy()
                else:
                    layer_acts_mean = np.zeros(n_neurons)
                activations[i, layer, :] = layer_acts_mean
    return activations

def save_activations(activations, path):
    """Save activations as a numpy file"""
    np.save(path, activations)

def load_activations(path):
    """Load activations from file"""
    return np.load(path)