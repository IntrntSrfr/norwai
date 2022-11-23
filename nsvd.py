import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import pathlib
import os
import matplotlib.pyplot as plt


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

class NSVD3(Dataset):
  def __init__(self, root, transforms=None, train=True) -> None:
    super(NSVD3, self).__init__()
    self.path = pathlib.Path(root) / 'NSVD' 
    self.df = pd.read_csv(self.path / 'data.csv')
    self.files = list((self.path / ('train' if train else 'test')).glob('*.jpg'))
    self.transforms = transforms
  
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, index):
    f = self.files[index]
    label = self.df.loc[self.df['filename'] == f.name].to_dict(orient='records')[0]["county"]
    img = Image.open(f)

    if self.transforms:
      img = self.transforms(img)
    return img, label

class NSVD_Boxes(Dataset):
  def __init__(self, root, transforms=None, train=True) -> None:
    super(NSVD_Boxes, self).__init__()
    self.path = pathlib.Path(root) / 'NSVD' 
    self.df = pd.read_csv(self.path / 'data_boxes.csv')
    self.df = self.df.groupby('box_index')
    print(self.df.size().mean())
    self.df = self.df.apply(lambda x: x.sample(int(self.df.size().mean()), replace=True).reset_index(drop=True))
    self.files = list((self.path / ('train' if train else 'test')).glob('*.jpg'))
    #Remove all files that are not in df
    self.files = [f for f in self.files if f.name in self.df['filename'].values]
    self.transforms = transforms
  
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, index):
    f = self.files[index]
    label = self.df.loc[self.df['filename'] == f.name].to_dict(orient='records')[0]
    img = Image.open(f)

    if self.transforms:
      img = self.transforms(img)
    return img, label

class NSVD4(Dataset):
  def __init__(self, root, train, label, return_coords, transforms=None):
    super(NSVD4, self).__init__()

    if label not in ['county', 'coords']: # add this later ", 'zone']:"
      raise ValueError('label must be one in ["county", "coords", "zone"]')
    self.label = label
    
    self.path = pathlib.Path(root) / 'NSVD' 
    self.df = pd.read_csv(self.path / 'data.csv')
    self.files = list((self.path / ('train' if train else 'test')).glob('*.jpg'))

    self.transforms = transforms
    self.return_coords = return_coords
  
  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    f = self.files[index]
    row = self.df.loc[self.df['filename'] == f.name]
    if self.label == 'coords':
      label = self.df.loc[self.df['filename'] == f.name][['lat', 'lng']].to_numpy()[0]
    else:
      label = self.df.loc[self.df['filename'] == f.name][[self.label]].to_numpy()[0][0]
    img = Image.open(f)
    if self.transforms:
      img = self.transforms(img)
    
    if self.return_coords:
      return img, label, row[['lat', 'lng']].to_numpy()[0]
    return img, label, -1


def plot_img(t):
  t = t.numpy().transpose((1,2,0))
  plt.imshow(t)
  plt.show()

if __name__ == '__main__':
  from torchvision import transforms
  import random

  tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
  ])

  d = NSVD4('data', True, "county", True, transforms=tf)

  img, label, coords = d[random.randint(0, len(d)-1)]
  print(label)
  print(coords)
  plot_img(img)