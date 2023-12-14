import pandas as pd
import re
from sklearn.model_selection import train_test_split

csv_file_path = 'data.txt'
csv_df = pd.read_csv(csv_file_path)

df_shuffled = csv_df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, test_df = train_test_split(df_shuffled, test_size=0.2, random_state=42)

train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)