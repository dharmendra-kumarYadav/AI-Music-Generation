import pandas as pd
import numpy as np
import re

def fix_array_syntax(data):
    pattern = r'\[\s*(\d+\s*)+\]'
    def fix_array(match):
        numbers = re.findall(r'\d+', match.group())
        return '[' + ' '.join(numbers) + ']'
    return re.sub(pattern, fix_array, data)

def augment_data(df, x_col_name):
    augmented_data = []
    for index, row in df.iterrows():
        original_notes = row[x_col_name]
        future_notes = row['future']
        augmented_data.append({x_col_name: original_notes, 'future': future_notes})
        
        original_notes_fixed = fix_array_syntax(original_notes)
        augmented_data.append({x_col_name: original_notes_fixed, 'future': future_notes})
    return pd.DataFrame(augmented_data)

df_tr = pd.read_csv('trainset.csv')
df_val = pd.read_csv('validationset.csv')
df_test = pd.read_csv('testset.csv')

augmented_tr = augment_data(df_tr, 'x_tr')
augmented_val = augment_data(df_val, 'x_val')
augmented_test = augment_data(df_test, 'x_test')

df_tr_augmented = pd.concat([df_tr, augmented_tr], ignore_index=True)
df_val_augmented = pd.concat([df_val, augmented_val], ignore_index=True)
df_test_augmented = pd.concat([df_test, augmented_test], ignore_index=True)

df_tr_augmented.to_csv('trainset_augmented.csv', index=False)
df_val_augmented.to_csv('validationset_augmented.csv', index=False)
df_test_augmented.to_csv('testset_augmented.csv', index=False)

print("Data augmentation completed and saved successfully!")
