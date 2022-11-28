import pandas as pd
import torch
from pyproj import Proj, transform
from torch import nn
import numpy as np
import geopy.distance

import warnings
warnings.filterwarnings("ignore")

#Get data files
n_list = pd.read_csv('data/NSVD/zone_boxesx200.csv')
gps = Proj(init='epsg:4326')
zone34N = Proj(init='epsg:23034')

#Box indexes used to get the original box coordinates
BOX_INDEXES = [43, 75, 22, 11, 31, 64, 77, 23, 33, 53, 21, 54, 76, 32, 42, 74, 86, 12, 87, 85]
EPOCHS = 13
MODEL = "EfficientNetS"
LR = 0.001
DATA_NORM = "avg" # "avg" or "min"

# Calculate centroid by creating a line and selecting the point x% into that line.
def weighted_centroid(points, weights):
    res = points
    while len(res) > 1:
        new_res = []
        new_weights = []
        for i in range(0,len(res),2):
            if i+1 == len(res):
              continue
            p1 = res[i]
            p2 = res[i+1]
            w1 = weights[i]
            w2 = weights[i+1]
            
            vector = np.array(p2) - np.array(p1)
            
            percentage = w1 / (w1 + w2)
            p = p1 + percentage*vector

            new_res.append(p)
            new_weights.append(w1 + w2)
        res = new_res
        weights = new_weights
    return res

# Calculate centroid 
def centroid(points):
  if type(points[0]) == str:
    points = [tuple(map(float, point[1:-1].split(','))) for point in points]

  x = [p[0] for p in points]
  y = [p[1] for p in points]
  return (sum(x) / len(points), sum(y) / len(points))
  
# Calculate distance between label points and predicted points
def distance_from_point(y, labels, amt = 3): 
  dist = []
  for i in range(len(y)):
    # Get the four best y values with in each batch
    y_ = y[i].topk(amt, dim=0)[1]
    lat, lng = labels["lat"][i], labels["lng"][i]
    # Get y_ to the cpu
    y_ = y_.cpu().numpy()
    # Get the box cords for each of the four best y values
    box_cords = n_list.iloc[[BOX_INDEXES[y_[idx]] for idx in range(len(y_))]]
  
    box_centers = []
    for _, row in box_cords.iterrows():
      box_centers.append(centroid([row['1'], row['0']]))
    
    percentages =  torch.softmax(y[i], dim=0).detach().cpu().numpy()
    #get percentage of each box selected
    percentages = percentages[[y_[idx] for idx in range(len(y_))]]
    #calculate new percentages so that they add up to 1
    percentages = percentages / percentages.sum()

    estimated_point = weighted_centroid(box_centers,percentages)[0]
    esitmated_point = transform(zone34N, gps, estimated_point[0], estimated_point[1])
    estimated_point = (esitmated_point[1], esitmated_point[0])
    dist.append(geopy.distance.geodesic((lat, lng), estimated_point).km)
  
  return dist
    
if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from torchvision import datasets, models, transforms
  from tqdm import tqdm, trange

  from nsvd import NSVD_Boxes
  from norwai_models import get_model

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  tf = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  train = NSVD_Boxes('./data', transforms=tf, data_normalization=DATA_NORM)
  train_ldr = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
  test = NSVD_Boxes('./data', train=False, transforms=tf, data_normalization=DATA_NORM)
  test_ldr = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)

  model = get_model(MODEL, len(BOX_INDEXES), False)

  model.to(device)
  
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
  
  losses = []
  accs = []

  for epoch in range(EPOCHS):
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
      epoch_loss += loss.item()
      pbar.set_description(" loss: {:.5f}".format(epoch_loss/(batch_idx + 1)))
      pbar.update()
    losses.append(epoch_loss/len(train_ldr))
    pbar.close()
    scheduler.step()
  
    pbar = trange(len(test_ldr), ascii=True)
    with torch.no_grad():
      epoch_acc = 0
      for batch_idx, (batch, labels) in enumerate(test_ldr):
        batch, labels = batch.to(device), labels
        y = model(batch)
        
        avd_dist = distance_from_point(y, labels)
        epoch_acc += np.mean(avd_dist)
        pbar.set_description("test:  epoch: {:2d};  acc: {:.5f}".format(epoch, (epoch_acc/(batch_idx + 1))))
        pbar.update()
    accs.append(epoch_acc)
    pbar.close()

  # Save the model 
  torch.save(model, './data/trained_models/norwai_zone.model')

  import matplotlib.pyplot as plt  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
  fig.set_facecolor('white')
  ax1.set_yscale('log')
  ax1.plot(range(len(losses)), losses, label='loss')
  ax1.legend()
  ax2.plot(range(len(accs)), accs, label='accuracy')
  ax2.legend()
  plt.show()