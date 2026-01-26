"""
Runs ablation experiments on virtue ethics evaluation dataset.
Tests multiple ablation strategies with matched random controls.
- All virtue neurons (including shared with other frameworks)
- Virtue-only neurons
- Top 10% virtue-only neurons by effect size
- Control for each group with random neurons matched by layer
"""

"""
NOTE - Saved few shot prompt that was removed from main, just in case
few_shot_prompt = Choose between Option A and Option B:

Context: The sky appears blue during the day.
Question: Therefore, the color of the sky is
Option A: blue
Option B: green
Answer: Option A

Context: Water freezes at 0 degrees Celsius.
Question: Therefore, at -5°C water will be
Option A: liquid
Option B: solid
Answer: Option B

"""
import torch
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import config
from tqdm import tqdm
from identify_virtue_neurons import identify_virtue_neurons

class AblationEvaluator:
    def __init__(self, model_name="gpt2"):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

         # quick sanity: ensure this is same GPT-2 small (12 layers, 768 hidden)
        cfg = self.model.config
        print(f"Loaded model: {getattr(cfg,'_name_or_path','unknown')}, n_layer={cfg.n_layer}, hidden_size={cfg.hidden_size}")
        assert cfg.n_layer == 12 and cfg.hidden_size == 768, "Loaded model does not look like GPT-2 small"
        
        # Load pre-identified virtue neurons from CSV
        self.virtue_neurons = pd.read_csv(os.path.join(config.STATS_DIR, 'virtue_ablation_neurons.csv'))
        self.neuron_groups = {}
        self._prepare_neuron_groups()
    
    def _prepare_neuron_groups(self):
        """Organize neurons by group and layer"""
        for group in ['all_virtue', 'virtue_only', 'top_virtue']:
            group_neurons = self.virtue_neurons[self.virtue_neurons['group'] == group]
            self.neuron_groups[group] = {}
            for layer in range(12):
                layer_neurons = group_neurons[group_neurons['layer'] == layer]['neuron_index'].tolist()
                self.neuron_groups[group][layer] = layer_neurons
    
    def _create_random_control(self, target_group):
        """Create random control neurons matching target group's layer distribution"""
        target_neurons = self.neuron_groups[target_group]
        random_control = {}
        
        for layer, neurons in target_neurons.items():
            n_neurons = len(neurons)
            if n_neurons > 0:
                # Randomly select neurons from this layer, excluding the target neurons
                all_neurons = set(range(3072))
                available_neurons = list(all_neurons - set(neurons))
                random_control[layer] = np.random.choice(
                    available_neurons, size=n_neurons, replace=False
                ).tolist()
            else:
                random_control[layer] = []
        
        return random_control
    
    def _ablate_forward(self, input_ids, ablation_neurons=None):
        """Run model forward pass with optional ablation"""
        if ablation_neurons is None:
            return self.model(input_ids)
        
        # Hook function for ablation
        def ablation_hook(module, input, output):
            if hasattr(module, 'current_layer'):
                layer = module.current_layer
                if layer in ablation_neurons and len(ablation_neurons[layer]) > 0:
                    # Zero out specified neurons
                    output[:, :, ablation_neurons[layer]] = 0
            return output
        
        # Register hooks
        hooks = []
        for i, layer in enumerate(self.model.transformer.h):
            layer.mlp.c_fc.current_layer = i
            hook = layer.mlp.c_fc.register_forward_hook(ablation_hook)
            hooks.append(hook)
        
        try:
            outputs = self.model(input_ids)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return outputs
    
    def evaluate_baseline(self, eval_df):
        """Run evaluation without ablation to get baseline"""
        print("Running baseline evaluation...")
        baseline_results = self._evaluate_group(eval_df, ablation_neurons=None, group_name="baseline")
        return baseline_results
    
    def evaluate_ablation_group(self, eval_df, group_name, baseline_results):
        """Evaluate a specific ablation group and its random control"""
        print(f"\nEvaluating {group_name} ablation...")
        
        # Main ablation
        ablation_neurons = self.neuron_groups[group_name]
        ablation_results = self._evaluate_group(eval_df, ablation_neurons, group_name)
        
        # Random control
        random_control = self._create_random_control(group_name)
        control_results = self._evaluate_group(eval_df, random_control, f"control_{group_name}")
        
        return ablation_results, control_results
    
    def _evaluate_group(self, eval_df, ablation_neurons, group_name):
        """Evaluate a single group"""
        results = []
        
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=group_name):
            # Prepare input
            input_text = f"{row['context_text']} {row['question_text']}"
            target_answer = row['target_answer']
            
            # Tokenize
            inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                outputs = self._ablate_forward(inputs['input_ids'], ablation_neurons)
                logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Get probabilities for "Yes" and "No"
            yes_token_id = self.tokenizer.encode("Yes", add_special_tokens=False)[0]
            no_token_id = self.tokenizer.encode("No", add_special_tokens=False)[0]
            
            yes_logit = logits[0, yes_token_id].item()
            no_logit = logits[0, no_token_id].item()
            
            # Convert to probabilities
            probs = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)
            yes_prob, no_prob = probs[0].item(), probs[1].item()
            
            # Determine prediction
            predicted_answer = "Yes" if yes_prob > no_prob else "No"
            is_correct = predicted_answer == target_answer
            
            results.append({
                'pair_id': row['pair_id'],
                'context_type': row['context_type'],
                'target_answer': target_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'yes_logit': yes_logit,
                'no_logit': no_logit,
                'yes_prob': yes_prob,
                'no_prob': no_prob,
                'group': group_name
            })
        
        return pd.DataFrame(results)
    
    def run_full_evaluation(self, eval_csv_path):
        """Run complete ablation evaluation"""
        # Load evaluation data
        eval_df = pd.read_csv(eval_csv_path)
        
        # Get baseline
        baseline_results = self.evaluate_baseline(eval_df)
        
        all_results = [baseline_results]
        
        # Test each ablation group
        for group in ['all_virtue', 'virtue_only', 'top_virtue']:
            ablation_results, control_results = self.evaluate_ablation_group(
                eval_df, group, baseline_results
            )
            all_results.extend([ablation_results, control_results])
        
        # Combine all results
        final_results = pd.concat(all_results, ignore_index=True)
        
        # Save results
        output_path = os.path.join(config.STATS_DIR, 'ablation_evaluation_results.csv')
        final_results.to_csv(output_path, index=False)
        
        # Generate summary statistics
        self._generate_summary(final_results)
        
        return final_results
    
    def _generate_summary(self, results_df):
        """Generate summary statistics and flip analysis"""
        summary_data = []
        
        for group in results_df['group'].unique():
            group_data = results_df[results_df['group'] == group]
            
            # Overall accuracy
            accuracy = group_data['is_correct'].mean()
            
            # Accuracy by context type
            moral_accuracy = group_data[group_data['context_type'] == 'moral']['is_correct'].mean()
            neutral_accuracy = group_data[group_data['context_type'] == 'neutral']['is_correct'].mean()
            
            # Count flips (if we had baseline comparison)
            if group != 'baseline':
                baseline_data = results_df[results_df['group'] == 'baseline']
                flip_count = self._count_flips(group_data, baseline_data)
            else:
                flip_count = 0
            
            summary_data.append({
                'group': group,
                'overall_accuracy': accuracy,
                'moral_accuracy': moral_accuracy,
                'neutral_accuracy': neutral_accuracy,
                'flip_count': flip_count,
                'n_samples': len(group_data)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(config.STATS_DIR, 'ablation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print("\nAblation Summary:")
        print(summary_df.to_string(index=False))
    
    def _count_flips(self, group_data, baseline_data):
        """Count answer flips compared to baseline"""
        flips = 0
        for (_, group_row), (_, baseline_row) in zip(group_data.iterrows(), baseline_data.iterrows()):
            if group_row['predicted_answer'] != baseline_row['predicted_answer']:
                flips += 1
        return flips

if __name__ == "__main__":
    evaluator = AblationEvaluator()

    eval_csv_path = os.path.join(config.DATA_DIR, 'virtue_eval.csv')
    
    results = evaluator.run_full_evaluation(eval_csv_path)
    print("Ablation evaluation complete!")