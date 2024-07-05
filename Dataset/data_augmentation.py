import pandas as pd
import numpy as np
import re

def fix_array_syntax(data):
    # Regular expression pattern to match arrays with incorrect syntax
    pattern = r'\[\s*(\d+\s*)+\]'
    # Function to fix the syntax of arrays
    def fix_array(match):
        # Extract the numbers from the matched string
        numbers = re.findall(r'\d+', match.group())
        # Join the numbers with spaces and surround with square brackets
        return '[' + ' '.join(numbers) + ']'
    # Replace arrays with incorrect syntax using the fix_array function
    return re.sub(pattern, fix_array, data)

def augment_data(df, x_col_name):
    augmented_data = []
    for index, row in df.iterrows():
        original_notes = row[x_col_name]
        future_notes = row['future']
        # Apply augmentation techniques
        # Example: Adding Gaussian noise to the notes
        augmented_data.append({x_col_name: original_notes, 'future': future_notes})
        
        # Fix the syntax errors in the original_notes array
        original_notes_fixed = fix_array_syntax(original_notes)
        augmented_data.append({x_col_name: original_notes_fixed, 'future': future_notes})
    return pd.DataFrame(augmented_data)

# Read the datasets
df_tr = pd.read_csv('trainset.csv')
df_val = pd.read_csv('validationset.csv')
df_test = pd.read_csv('testset.csv')

# Apply augmentation to each dataset
augmented_tr = augment_data(df_tr, 'x_tr')
augmented_val = augment_data(df_val, 'x_val')
augmented_test = augment_data(df_test, 'x_test')

# Concatenate augmented data with original data
df_tr_augmented = pd.concat([df_tr, augmented_tr], ignore_index=True)
df_val_augmented = pd.concat([df_val, augmented_val], ignore_index=True)
df_test_augmented = pd.concat([df_test, augmented_test], ignore_index=True)

# Save the increased datasets
df_tr_augmented.to_csv('trainset_augmented.csv', index=False)
df_val_augmented.to_csv('validationset_augmented.csv', index=False)
df_test_augmented.to_csv('testset_augmented.csv', index=False)

print("Data augmentation completed and saved successfully!")
