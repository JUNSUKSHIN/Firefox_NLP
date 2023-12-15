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

row = """index  | sentence                                        | flag
---------------------------------------------------------
0      | 오늘 밤의 별자리를 구글에 검색해줘서 어떤 별자리가 보이는지 확인해보자. | 1
1      | 내가 좋아하는 그 가수의 뮤직비디오 애플 TV에서 찾아보면 좋겠다.       | 0
2      | 애플 TV에서 최신 공포 영화를 검색해줘.                           | 0
3      | 넷플릭스에서 검색한 영화 목록을 내게 보내줄 수 있을까.           | 2
4      | 넷플릭스에서 검색하면 어떤 영화가 나오는지 알려줘.               | 2"""

print(row)