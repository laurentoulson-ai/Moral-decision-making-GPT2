"""The orginal virtue_eval.csv format is not compatible with how GPT-2 expects to produce answers and therefore did not work as expected
This script converts the original eval questions into a forced-choice format with option_a and option_b to enable better evaluation."""

import pandas as pd

def convert_virtue_eval_format(input_file, output_file):
    """
    Convert the virtue_eval.csv format to the new format with option_a and option_b
    """
    # Read the original CSV file
    df = pd.read_csv(input_file)
    
    # Create a new dataframe for the converted format
    converted_data = []
    
    # Group by pair_id
    for pair_id in sorted(df['pair_id'].unique()):
        pair_data = df[df['pair_id'] == pair_id]
        
        # Separate moral and neutral rows
        moral_rows = pair_data[pair_data['context_type'] == 'moral']
        neutral_rows = pair_data[pair_data['context_type'] == 'neutral']
        
        # Check if we have enough rows
        if len(moral_rows) < 2 or len(neutral_rows) < 2:
            print(f"Warning: pair_id {pair_id} has insufficient rows. Skipping...")
            continue
        
        # Get moral context (first moral entry)
        moral_context = moral_rows.iloc[0]['context_text']
        
        # Get neutral context (first neutral entry)
        neutral_context = neutral_rows.iloc[0]['context_text']
        
        def extract_action(question_text):
            """Extract the action part from question text, removing pronouns"""
            # Remove the "Therefore, should " prefix and "?" suffix
            action = question_text.replace('Therefore, should ', '').replace('?', '').strip()
            
            # Remove leading pronouns (he/she) and any following space
            if action.startswith('he '):
                action = action[3:]  # Remove "he "
            elif action.startswith('she '):
                action = action[4:]  # Remove "she "
            
            return action
        
        # Extract options from moral rows
        moral_options = []
        for _, row in moral_rows.iterrows():
            action = extract_action(row['question_text'])
            moral_options.append(action)
        
        # Extract options from neutral rows
        neutral_options = []
        for _, row in neutral_rows.iterrows():
            action = extract_action(row['question_text'])
            neutral_options.append(action)
        
        # Create entries for moral and neutral contexts
        moral_entry = {
            'pair_id': pair_id,
            'context_type': 'moral',
            'context_text': moral_context,
            'question_text': 'Therefore, he/she should',
            'option_a': moral_options[0],  # First moral option (e.g., "disclose...")
            'option_b': moral_options[1]   # Second moral option (e.g., "conceal...")
        }
        
        neutral_entry = {
            'pair_id': pair_id,
            'context_type': 'neutral',
            'context_text': neutral_context,
            'question_text': 'Therefore, he/she should',
            'option_a': neutral_options[0],  # First neutral option
            'option_b': neutral_options[1]   # Second neutral option
        }
        
        converted_data.extend([moral_entry, neutral_entry])
    
    # Create new dataframe
    converted_df = pd.DataFrame(converted_data)
    
    # Save to new CSV file
    converted_df.to_csv(output_file, index=False)
    
    print(f"\nConversion complete! Converted {len(converted_data)} rows.")
    print(f"Original had {len(df)} rows, new format has {len(converted_data)} rows.")
    print(f"Processed {len(converted_data)//2} pair_ids successfully.")
    
    return converted_df

# Usage
if __name__ == "__main__":
    input_filename = "virtue_eval.csv"
    output_filename = "virtue_eval_converted.csv"
    
    converted_data = convert_virtue_eval_format(input_filename, output_filename)
    
    # Display first few rows to verify
    print("\nFirst few rows of converted data:")
    print(converted_data.head(4))