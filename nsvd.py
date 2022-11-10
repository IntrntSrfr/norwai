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
''' 
if __name__ == '__main__':
  train = NSVD()
  print(train.targets.shape)
  print(train.data.shape)
 '''
#With county
class NSVD_COUNTY(Dataset):
  def __init__(self):
    self.df = pd.read_csv('./data/locations_county.csv').iloc[:3600]
    lst = os.listdir("./data/images")
    self.df = self.df[self.df['filename'].str.contains('|'.join(lst)) == True]
    self.targets = torch.tensor(self.df["fylkesnummer"].to_numpy()).long()
    self.data = torch.tensor(np.array([self[idx] for idx in range(len(self.df))])).float()#torch.tensor([self[idx] for idx in range(5)])
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    d = self.df.iloc[[idx]]
    image = Image.open(os.path.join('./data/images', d['filename'].item())).resize((256, 256))
    return np.asarray(image).transpose(2, 0, 1)

if __name__ == '__main__':
  train = NSVD_COUNTY()
  print(train.data.shape)
  print(train.data.mean(dim=[0, 2, 3]))
  #print(train.targets)
  #print(train.data.shape)

import pathlib

class NSVD2(Dataset):
  def __init__(self, root, transforms=None, train=True) -> None:
    super(NSVD2, self).__init__()
    self.path = pathlib.Path(root) / 'NSVD' / ('train' if train else 'test')
    self.files = list(self.path.glob('*/*.jpg'))
    self.classes = [d.name for d in self.path.iterdir() if d.is_dir()]
    self.idx_to_class = {i:v for i, v in enumerate(self.classes)}
    self.class_to_idx = {v:i for i, v in enumerate(self.classes)}

    self.transforms = transforms

  def __len__(self):
    return len(self.files)

  def __getitem__(self, idx):
    f = self.files[idx]
    label = self.class_to_idx[f.parent.name]
    img = Image.open(f)

    if self.transforms:
      img = self.transforms(img)
    return img, label
  