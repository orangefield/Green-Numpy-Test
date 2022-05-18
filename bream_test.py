from operator import index
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# csv 파일 읽기
bream_length = pd.read_csv('bream_length.csv')
bream_weight = pd.read_csv('bream_weight.csv')
smelt_length = pd.read_csv('smelt_length.csv')
smelt_weight = pd.read_csv('smelt_weight.csv')


# 도미와 빙어 데이터 시각화
# plt.scatter(bream_length, bream_weight)
# plt.scatter(smelt_length, smelt_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 데이터프레임을 넘파이로 변환
b_length = bream_length.to_numpy()
b_weight = bream_weight.to_numpy()
s_length = smelt_length.to_numpy()
s_weight = smelt_weight.to_numpy()


# print(b_length)  # 왜 1번 데이터가 사라지지?????
# print(s_length)
# print(len(b_length))  # 왜 34개?
# print(len(b_weight))
# print(len(s_length))  # 왜 13개????
# print(len(s_weight))


# 넘파이 배열 합치기
length = np.concatenate([b_length, s_length])  # 47개 왜???
weight = np.concatenate([b_weight, s_weight])
fish_data = np.column_stack((length, weight))

# 타겟
fish_target = np.array([1]*34 + [0]*13)


# 두 배열 동시에 섞기
index = np.arange(fish_data.shape[0])
np.random.shuffle(index)

shuffled_fish_data = fish_data[index]
shuffled_fish_target = fish_target[index]

# print(shuffled_fish_data)
# print("="*50)
# print(shuffled_fish_target)


# 훈련데이터, 테스트데이터 나누기
train_input, test_input, train_target, test_target = train_test_split(
    shuffled_fish_data, shuffled_fish_target)
# print(train_input.shape)  # (35, 2)
# print(test_input.shape)  # (12, 2)
# print(train_input)
# print(train_target)


# 도미와 빙어 데이터 시각화
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
