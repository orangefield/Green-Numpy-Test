import numpy as np
import pandas as pd


bream_length = pd.read_csv('bream_length.csv')
smelt_length = pd.read_csv('smelt_length.csv')

b_length = bream_length.to_numpy()
s_length = smelt_length.to_numpy()
length = np.concatenate([b_length, s_length])
print(length)

fish_target = [1]*34 + [0]*13
# print(fish_target)
print("="*50)

np.random.shuffle(length)
print(length)
