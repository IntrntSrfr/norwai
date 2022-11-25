from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

IDX2COUNTY = {
  0: 'agder',
  1: 'troms og finnmark',
  2: 'møre og romsdal',
  3: 'vestfold og telemark',
  4: 'trøndelag',
  5: 'rogaland',
  6: 'innlandet',
  7: 'viken',
  8: 'nordland',
  9: 'vestland',
}

class NSVD(Dataset):
  def __init__(self, root, train, label, return_coords, transforms=None):
    super(NSVD, self).__init__()

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


class NSVD_B(Dataset):
  def __init__(self, root, train, label, return_coords, transforms=None):
    super(NSVD_B, self).__init__()

    if label not in ['county', 'coords']: # add this later ", 'zone']:"
      raise ValueError('label must be one in ["county", "coords", "zone"]')
    self.label = label
    
    self.path = pathlib.Path(root) / 'NSVD_B' 
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


class NSVD_Boxes(Dataset):
  def __init__(self, root, transforms=None, zone_size="x200", data_normalization="min",  train=True) -> None:
    super(NSVD_Boxes, self).__init__()
    self.path = pathlib.Path(root) / 'NSVD' 
    self.df = pd.read_csv(self.path / 'data_boxes{}.csv'.format(zone_size))
    
    # Group boxes
    self.df = self.df.groupby('box_index')
    if data_normalization == 'min': 
      self.data_normalization = self.df.size().min() 
    else:
      self.data_normalization = self.df.size().mean() 
    self.df = self.df.apply(lambda x: x.sample(int(self.data_normalization), replace=True).reset_index(drop=True))
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

  d = NSVD('data', True, "county", True, transforms=tf)

  img, label, coords = d[random.randint(0, len(d)-1)]
  print(label)
  print(coords)
  plot_img(img)