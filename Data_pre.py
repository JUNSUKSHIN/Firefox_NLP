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

# 'target' 열의 빈도 수 계산
target_counts = train_df['target'].value_counts()

# 막대 그래프로 시각화
target_counts.plot(kind='bar')
plt.title('Target Value Counts')
plt.xlabel('Target')
plt.ylabel('Counts')
plt.show()