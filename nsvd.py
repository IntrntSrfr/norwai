import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import os

class NSVD(Dataset):
  def __init__(self):
    self.df = pd.read_csv('./data/locations.csv').iloc[:3600]
    self.targets = torch.tensor(self.df[['lat', 'lng']].to_numpy()).float()
    self.data = torch.tensor(np.array([self[idx] for idx in range(len(self.df))])).float()#torch.tensor([self[idx] for idx in range(5)])
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    d = self.df.iloc[[idx]]
    image = Image.open(os.path.join('./data/images', d['filename'].item())).resize((128, 128))
    return np.asarray(image).transpose(2, 0, 1)

if __name__ == '__main__':
  train = NSVD()
  print(train.targets.shape)
  print(train.data.shape)

#With county
class NSVD_COUNTY(Dataset):
  def __init__(self):
    self.df = pd.read_csv('./data/locations_county.csv').iloc[:3600]
    self.targets = torch.tensor(self.df["fylkesnummer"].to_numpy()).long()
    self.data = torch.tensor(np.array([self[idx] for idx in range(len(self.df))])).float()#torch.tensor([self[idx] for idx in range(5)])
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    d = self.df.iloc[[idx]]
    image = Image.open(os.path.join('./data/images', d['filename'].item())).resize((128, 128))
    return np.asarray(image).transpose(2, 0, 1)

if __name__ == '__main__':
  train = NSVD_COUNTY()
  print(train.targets.shape)
  print(train.data.shape)