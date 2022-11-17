import pandas as pd
import torch
from pyproj import Proj, transform
from torch import nn
import numpy as np
import geopy.distance

import warnings
warnings.filterwarnings("ignore")

#read n_list from n_list.csv
n_list = pd.read_csv('data/NSVD/n_list.csv')

gps = Proj(init='epsg:4326')
zone34N = Proj(init='epsg:23034')

def centroid(points):
  #If list of strings
  #Make list of string containing a float tuple into float tuples
  if type(points[0]) == str:
    points = [tuple(map(float, point[1:-1].split(','))) for point in points]

  x = [p[0] for p in points]
  y = [p[1] for p in points]
  centroid = (sum(x) / len(points), sum(y) / len(points))
  return centroid

def distance_from_point(y, labels): 
  dist = []
  BOX_INDEXES = [61, 18, 9, 35, 19, 43, 17, 44, 25, 62, 52, 27, 26, 10, 34, 60, 36]
  for i in range(len(y)):
    # Get the four best y values with in each batch
    y_ = y[i].topk(1, dim=0)[1]
    label_, lat, lng = labels["box_index"][i], labels["lat"][i], labels["lng"][i]
    # Get y_ to the cpu
    y_ = y_.cpu().numpy()
    # Get the box cords for each of the four best y values
    box_cords = n_list.iloc[y_]
    
    box_centers = []
    for i, row in box_cords.iterrows():
      box_centers.append(centroid([row['0'], row['1']]))
    
    estimated_point = centroid(box_centers)

    # Transform the center lat lng to zone34N
    #actual_point = transform(gps, zone34N, lng, lat)

    # Calculate the distance between the estimated point and the actual point
    dist.append(geopy.distance.geodesic((lat, lng), transform(zone34N, gps, estimated_point[1], estimated_point[0])).km)

  idx = dist.index(min(dist))
  print(BOX_INDEXES[labels["box_index"][idx]], labels["lat"][idx], labels["lng"][idx], min(dist))

  return dist

    

class ConvModule(nn.Module):
  def __init__(self, in_features, out_features) -> None:
    super(ConvModule, self).__init__()
    self.conv = nn.Conv2d(in_features, out_features, 5, padding=2)
    self.norm = nn.BatchNorm2d(out_features)

  def forward(self, x) -> torch.Tensor:
    x = self.norm(self.conv(x))
    return torch.relu(x)

class NSVDModel(nn.Module):
  def __init__(self) -> None:
    super(NSVDModel, self).__init__()
    self.pool = nn.MaxPool2d(2)
    self.conv1 = ConvModule(3, 64)
    self.conv2 = ConvModule(64, 128)
    self.conv3 = ConvModule(128, 256)
    self.conv4 = ConvModule(256, 256*2)
    #self.conv5 = ConvModule(256*2, 256*4)
    #self.conv6 = ConvModule(256*4, 256*8)
    self.l1 = nn.Linear(256*8*8*2, 512*2)
    self.l2 = nn.Linear(512*2, 17)
    
  def forward(self, x):
    x = self.pool(self.conv1(x)) # 128 -> 64
    x = self.pool(self.conv2(x)) # 64 -> 32
    x = self.pool(self.conv3(x)) # 32 -> 16
    x = self.pool(self.conv4(x)) # 16 -> 8
    #x = self.pool(self.conv5(x)) # 32 -> 16
    #x = self.pool(self.conv6(x)) # 16 -> 8
    x = x.reshape(-1, 256*8*8*2)
    x = torch.relu(self.l1(x))
    x = self.l2(x)
    return x

if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from torchvision import datasets, models, transforms
  from tqdm import tqdm, trange

  from nsvd import NSVD_Boxes, NSVD3

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  tf = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  train = NSVD_Boxes('./data', transforms=tf)
  train_ldr = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
  test = NSVD_Boxes('./data', train=False, transforms=tf)
  test_ldr = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)

  model = NSVDModel()
  """
  model = models.vgg11_bn(progress=False, weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
  for i, param in enumerate(model.features.parameters()):
    param.requires_grad = False
  model.classifier[6] = nn.Linear(4096, 11)
  """
  model.to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

  losses = []
  accs = []

  epochs = 10
  for epoch in range(epochs):
    print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
    epoch_loss = 0
    pbar = trange(len(train_ldr), ascii=True)
    for batch_idx, (batch, labels) in enumerate(train_ldr):
      batch, labels = batch.to(device), labels['box_index'].to(device)
      y = model(batch)
      loss = loss_fn(y, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss.item()/len(train_ldr)

      pbar.set_description("    loss: {:.5f}".format(epoch_loss))
      pbar.update()
    losses.append(epoch_loss)
    pbar.close()
    scheduler.step()

    pbar = trange(len(test_ldr), ascii=True)
    with torch.no_grad():
      epoch_acc = 0
      for batch_idx, (batch, labels) in enumerate(test_ldr):
        batch, labels = batch.to(device), labels
        y = model(batch)
        # for each batch, get the four highest values from y and calculate the distance from the point
        avd_dist = distance_from_point(y, labels)
        #y = torch.argmax(torch.exp(y), dim=1)
        #epoch_acc += (1-torch.count_nonzero(y-labels).item() / len(batch))/len(test_ldr)
        epoch_acc += np.mean(avd_dist)/len(test_ldr)
        pbar.set_description("test:  epoch: {:2d};  acc: {:.5f}".format(epoch, epoch_acc))
        pbar.update()
    accs.append(epoch_acc)
    pbar.close()
  torch.save(model, './data/model_classification')
"""
  import matplotlib.pyplot as plt  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
  fig.set_facecolor('white')
  ax1.set_yscale('log')
  ax1.plot(range(len(losses)), losses, label='loss')
  ax1.legend()
  ax2.plot(range(len(accs)), accs, label='accuracy')
  ax2.legend()
  plt.show()
"""
