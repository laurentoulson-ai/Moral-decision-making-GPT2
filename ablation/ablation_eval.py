"""
Runs ablation experiments on virtue ethics evaluation dataset.
Tests multiple ablation strategies with matched random controls.
- All virtue neurons (including shared with other frameworks)
- Virtue-only neurons
- Top 10% virtue-only neurons by effect size
- Control for each group with random neurons matched by layer
"""
import torch
import pandas as pd
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import json
import os
import config
from tqdm import tqdm

class ImprovedAblationEvaluator:
    def __init__(self, model_name="gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-identified virtue neurons (prefer specialisation stats)
        special_stats_dir = getattr(config, 'SPECIALISATION_STATS_DIR', None)
        if special_stats_dir and os.path.exists(special_stats_dir):
            virtue_neurons_path = os.path.join(special_stats_dir, 'virtue_ablation_neurons.csv')
        else:
            virtue_neurons_path = os.path.join(config.STATS_DIR, 'virtue_ablation_neurons.csv')

        if not os.path.exists(virtue_neurons_path):
            raise FileNotFoundError(f"Virtue neurons CSV not found at {virtue_neurons_path}")

        self.virtue_neurons = pd.read_csv(virtue_neurons_path)

        # Diagnostic: Check neuron counts
        print("\n=== NEURON GROUP SIZES ===")
        for group in self.virtue_neurons['group'].unique():
            count = len(self.virtue_neurons[self.virtue_neurons['group'] == group])
            print(f"{group}: {count} neurons")
        
        self.neuron_groups = {}
        self._prepare_neuron_groups()
    
    def _prepare_neuron_groups(self):
        """Organize neurons by group and layer with detailed logging"""
        for group in ['all_virtue', 'virtue_only', 'top_virtue']:
            group_neurons = self.virtue_neurons[self.virtue_neurons['group'] == group]
            self.neuron_groups[group] = {}
            total = 0
            
            print(f"\n{group} distribution:")
            for layer in range(12):
                layer_neurons = group_neurons[group_neurons['layer'] == layer]['neuron_index'].tolist()
                self.neuron_groups[group][layer] = layer_neurons
                total += len(layer_neurons)
                if len(layer_neurons) > 0:
                    print(f"  Layer {layer}: {len(layer_neurons)} neurons")
            print(f"  Total: {total} neurons")
    
    def _create_random_control(self, target_group):
        """Create random control neurons matching target group's layer distribution"""
        target_neurons = self.neuron_groups[target_group]
        random_control = {}
        
        for layer, neurons in target_neurons.items():
            n_neurons = len(neurons)
            if n_neurons > 0:
                all_neurons = set(range(3072))
                available_neurons = list(all_neurons - set(neurons))
                random_control[layer] = np.random.choice(
                    available_neurons, size=n_neurons, replace=False
                ).tolist()
            else:
                random_control[layer] = []
        
        return random_control
    
    def _apply_ablation_to_model(self, model, ablation_neurons):
        """Apply permanent ablation by zeroing neuron outputs"""
        total_ablated = 0
        for layer_idx, neurons in ablation_neurons.items():
            if neurons:
                with torch.no_grad():
                    mlp = model.transformer.h[layer_idx].mlp
                    mlp.c_proj.weight.data[neurons, :] = 0
                    mlp.c_fc.bias.data[neurons] = 0
                    total_ablated += len(neurons)
        
        print(f"  Ablated {total_ablated} neurons total")
        return total_ablated
    
    def _evaluate_group(self, eval_df, ablation_neurons, group_name):
        """Evaluate a single group - NO few-shot prompting"""
        results = []
        
        # Create and ablate model once for this group
        print(f"\nPreparing model for {group_name}...")
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        model.to(self.device)
        
        if ablation_neurons is not None:
            n_ablated = self._apply_ablation_to_model(model, ablation_neurons)
        
        model.eval()
        
        # Pre-compute token IDs (check both with and without space)
        token_a_space = self.tokenizer.encode(" A", add_special_tokens=False)[0]
        token_b_space = self.tokenizer.encode(" B", add_special_tokens=False)[0]
        token_a_no_space = self.tokenizer.encode("A", add_special_tokens=False)[0]
        token_b_no_space = self.tokenizer.encode("B", add_special_tokens=False)[0]
        
        # Print for first iteration only
        if group_name == "baseline":
            print(f"Token IDs: A(space)={token_a_space}, A(no space)={token_a_no_space}, B(space)={token_b_space}, B(no space)={token_b_no_space}")
        
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=group_name):
            # Simple prompt - no few-shot examples
            prompt = f"Context: {row['context_text']}\nQuestion: {row['question_text']}\nOption A: {row['option_a']}\nOption B: {row['option_b']}\nAnswer: Option"
            
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(inputs['input_ids'])
                    next_token_logits = outputs.logits[0, -1, :]
                    log_probs = torch.log_softmax(next_token_logits, dim=0)
                    
                    # Get logprobs for both versions and use the higher one
                    logprob_a = max(log_probs[token_a_space].item(), log_probs[token_a_no_space].item())
                    logprob_b = max(log_probs[token_b_space].item(), log_probs[token_b_no_space].item())
                
                prefers_a = logprob_a > logprob_b
                
                # Check against the actual correct answer
                if 'correct_answer' in row:
                    is_correct = (prefers_a and row['correct_answer'] == 'option_a') or \
                                (not prefers_a and row['correct_answer'] == 'option_b')
                else:
                    # Fallback: assume A is always correct (for backwards compatibility)
                    is_correct = prefers_a
                
                results.append({
                    'pair_id': row['pair_id'],
                    'context_type': row['context_type'],
                    'predicted_answer': 'option_a' if prefers_a else 'option_b',
                    'correct_answer': row.get('correct_answer', 'option_a'),
                    'is_correct': is_correct,
                    'logprob_a': logprob_a,
                    'logprob_b': logprob_b,
                    'logprob_diff': logprob_a - logprob_b,
                    'group': group_name
                })
                
            except Exception as e:
                print(f"Error on row {row['pair_id']}: {e}")
                continue
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return pd.DataFrame(results)
    
    def run_full_evaluation(self, eval_csv_path):
        """Run complete ablation evaluation with diagnostics"""
        eval_df = pd.read_csv(eval_csv_path)
        
        print(f"\n=== EVALUATION DATASET ===")
        print(f"Total samples: {len(eval_df)}")
        print(f"Moral contexts: {(eval_df['context_type'] == 'moral').sum()}")
        print(f"Neutral contexts: {(eval_df['context_type'] == 'neutral').sum()}")
        
        # Baseline
        print("\n=== BASELINE ===")
        baseline_results = self._evaluate_group(eval_df, None, "baseline")
        
        all_results = [baseline_results]
        
        # Test each ablation group
        for group in ['all_virtue', 'virtue_only', 'top_virtue']:
            print(f"\n=== {group.upper()} ===")
            ablation_results = self._evaluate_group(
                eval_df, self.neuron_groups[group], group
            )
            
            print(f"\n=== CONTROL_{group.upper()} ===")
            random_control = self._create_random_control(group)
            control_results = self._evaluate_group(
                eval_df, random_control, f"control_{group}"
            )
            
            all_results.extend([ablation_results, control_results])
        
        # Combine and save
        final_results = pd.concat(all_results, ignore_index=True)
        output_path = os.path.join(config.STATS_DIR, 'ablation_evaluation_results.csv')
        final_results.to_csv(output_path, index=False)
        
        self._generate_summary(final_results)
        return final_results
    
    def _generate_summary(self, results_df):
        """Generate detailed summary with diagnostics"""
        summary_data = []
        baseline_data = results_df[results_df['group'] == 'baseline'].reset_index(drop=True)
        
        for group in results_df['group'].unique():
            group_data = results_df[results_df['group'] == group].reset_index(drop=True)
            
            accuracy = group_data['is_correct'].mean()
            moral_acc = group_data[group_data['context_type'] == 'moral']['is_correct'].mean()
            neutral_acc = group_data[group_data['context_type'] == 'neutral']['is_correct'].mean()
            avg_ll_diff = group_data['logprob_diff'].mean()
            
            if group != 'baseline':
                flip_count = (group_data['predicted_answer'].values != baseline_data['predicted_answer'].values).sum()
            else:
                flip_count = 0
            
            summary_data.append({
                'group': group,
                'overall_accuracy': accuracy,
                'moral_accuracy': moral_acc,
                'neutral_accuracy': neutral_acc,
                'avg_logprob_diff': avg_ll_diff,
                'flip_count': flip_count,
                'n_samples': len(group_data)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(config.STATS_DIR, 'ablation_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        print("\n=== ABLATION SUMMARY ===")
        print(summary_df.to_string(index=False))
        
        # Additional diagnostic
        print("\n=== DIAGNOSTIC: Expected Effects ===")
        baseline_acc = summary_df[summary_df['group'] == 'baseline']['overall_accuracy'].values[0]
        for group in ['all_virtue', 'virtue_only', 'top_virtue']:
            ablation_acc = summary_df[summary_df['group'] == group]['overall_accuracy'].values[0]
            control_acc = summary_df[summary_df['group'] == f'control_{group}']['overall_accuracy'].values[0]
            
            print(f"\n{group}:")
            print(f"  Baseline: {baseline_acc:.3f}")
            print(f"  Ablation: {ablation_acc:.3f} (change: {ablation_acc - baseline_acc:+.3f})")
            print(f"  Control:  {control_acc:.3f} (change: {control_acc - baseline_acc:+.3f})")
            print(f"  ✓ Expect: ablation < control (targeted > random)" if ablation_acc < control_acc else "  ✗ WARNING: ablation >= control")

if __name__ == "__main__":
    evaluator = ImprovedAblationEvaluator()
    eval_csv_path = os.path.join(config.DATA_DIR, 'virtue_eval.csv')
    results = evaluator.run_full_evaluation(eval_csv_path)
    print("\nAblation evaluation complete!")