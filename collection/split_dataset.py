import numpy as np
import pandas as pd
import shutil
import pathlib
from tqdm import tqdm
import os

df = pd.read_csv('./data/data.csv')
files = os.listdir('./data/images')
df = df[df['filename'].isin(files)]

tests = np.random.choice(range(len(df)), int(len(df)*.15), replace=False)
test_df = df.loc[df.index.isin(tests)]
train_df = df.loc[~df.index.isin(tests)]

for i, row in tqdm(train_df.iterrows(), total=len(train_df)):
  src = pathlib.Path('./data/images') / row['filename']
  dst = pathlib.Path('./data/NSVD/train') / row['filename']
  shutil.copy(src, dst)

for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
  src = pathlib.Path('./data/images') / row['filename']
  dst = pathlib.Path('./data/NSVD/test') / row['filename']
  shutil.copy(src, dst)