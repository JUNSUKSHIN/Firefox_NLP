import pandas as pd
import re
import matplotlib.pyplot as plt

train = 'data.txt'

train_df = pd.read_csv(train)

print(train_df.info())
print(train_df.head())
