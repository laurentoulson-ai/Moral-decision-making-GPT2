"""
Fix the evaluation dataset to properly counterbalance options
"""
import pandas as pd
import numpy as np
import os
import config

# Load the original eval dataset
eval_df = pd.read_csv(os.path.join(config.DATA_DIR, 'virtue_eval.csv'))

print(f"Original dataset: {len(eval_df)} rows")
print(f"Unique pairs: {eval_df['pair_id'].nunique()}")

# Create a properly counterbalanced version
fixed_rows = []

for pair_id in eval_df['pair_id'].unique():
    # Get both moral and neutral contexts for this pair
    pair_data = eval_df[eval_df['pair_id'] == pair_id]
    
    # Randomly decide whether to flip the options for this pair
    flip = np.random.random() < 0.5
    
    for _, row in pair_data.iterrows():
        if flip:
            # Swap option A and B
            new_row = row.copy()
            new_row['option_a'] = row['option_b']
            new_row['option_b'] = row['option_a']
            new_row['correct_answer'] = 'option_b'  # Now B is correct
            fixed_rows.append(new_row)
        else:
            # Keep as is
            new_row = row.copy()
            new_row['correct_answer'] = 'option_a'  # A is correct
            fixed_rows.append(new_row)

fixed_df = pd.DataFrame(fixed_rows)

# Verify the distribution
print("\n=== VERIFICATION ===")
print(f"Fixed dataset: {len(fixed_df)} rows")
print(f"\nCorrect answer distribution:")
print(fixed_df['correct_answer'].value_counts())
print(f"\nCorrect answer by context type:")
print(pd.crosstab(fixed_df['context_type'], fixed_df['correct_answer']))

# Save the fixed dataset
output_path = os.path.join(config.DATA_DIR, 'virtue_eval_balanced.csv')
fixed_df.to_csv(output_path, index=False)
print(f"\nSaved balanced dataset to: {output_path}")

# Show some examples
print("\n=== SAMPLE BALANCED ROWS ===")
for i in range(4):
    row = fixed_df.iloc[i]
    correct_option = row['option_a'] if row['correct_answer'] == 'option_a' else row['option_b']
    print(f"\nPair {row['pair_id']} ({row['context_type']}):")
    print(f"  Context: {row['context_text'][:60]}...")
    print(f"  A: {row['option_a'][:50]}...")
    print(f"  B: {row['option_b'][:50]}...")
    print(f"  Correct: {row['correct_answer']} ({correct_option[:40]}...)")