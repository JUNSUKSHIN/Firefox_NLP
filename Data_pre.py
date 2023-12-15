import pandas as pd
import re
import matplotlib.pyplot as plt

train = 'data.csv'

train_df = pd.read_csv(train)

# 데이터프레임 정렬 및 출력
with pd.option_context('display.max_colwidth', -1):  
    print(train_df)

# 기본 정보 출력
print(train_df.info())
print(train_df.head())