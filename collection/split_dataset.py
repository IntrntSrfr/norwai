import numpy as np
import pandas as pd
import shutil
import pathlib
import os
df = pd.read_csv('./data/data.csv')
lst = os.listdir("./data/images")
df = df[df['filename'].str.contains('|'.join(lst)) == True]
tests = np.random.choice(range(len(df)), int(len(df)*.15), replace=False)
test_df = df.loc[df.index.isin(tests)]
train_df = df.loc[~df.index.isin(tests)]

for i, row in train_df.iterrows():
  src = pathlib.Path('./data/images') / row['filename']
  dst = pathlib.Path('./data/NSVD/train') / row['filename']
  shutil.copy(src, dst)

for i, row in test_df.iterrows():
  src = pathlib.Path('./data/images') / row['filename']
  dst = pathlib.Path('./data/NSVD/test') / row['filename']
  shutil.copy(src, dst)
