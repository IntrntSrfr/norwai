import pandas as pd
import torch
from pyproj import Proj, transform
from torch import nn
import numpy as np
import geopy.distance
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms
from norwai_county import get_model;
from nsvd import NSVD_Boxes
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

gps = Proj(init='epsg:4326')
zone34N = Proj(init='epsg:23034')

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
            #length = np.linalg.norm(vector)
            
            percentage = w1 / (w1 + w2)
            p = p1 + percentage*vector

            new_res.append(p)
            new_weights.append(w1 + w2)
        res = new_res
        weights = new_weights
    return res

def centroid(points):
  if type(points[0]) == str:
    points = [tuple(map(float, point[1:-1].split(','))) for point in points]

  x = [p[0] for p in points]
  y = [p[1] for p in points]
  return (sum(x) / len(points), sum(y) / len(points))

def distance_from_point(y, labels, amt = 3): 
  #grad_fn = y.grad_fn
  dist = []
  #read n_list from n_list.csv
  n_list = pd.read_csv('data/NSVD/zone_boxes{}.csv'.format(wandb.config.zone_size))

  BOX_INDEXES =  [43, 75, 22, 11, 31, 64, 77, 23, 33, 53, 21, 54, 76, 32, 42, 86, 12, 87] if wandb.config.zone_size == 'x200' else [14, 27, 7, 13, 20, 28, 8]
  #print(len(y))
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
    
    #print(torch.softmax(y[i], dim=0).detach().cpu().numpy())
    percentages =  torch.softmax(y[i], dim=0).detach().cpu().numpy()
    #get percentage of each box selected
    percentages = percentages[[y_[idx] for idx in range(len(y_))]]
    #calculate new percentages so that they add up to 1
    percentages = percentages / percentages.sum()

    estimated_point = weighted_centroid(box_centers,percentages)[0]
    #print(estimated_point)
    esitmated_point = transform(zone34N, gps, estimated_point[0], estimated_point[1])
    estimated_point = (esitmated_point[1], esitmated_point[0])
    #y[i] = geopy.distance.distance(estimated_point, (lat, lng)).km
    dist.append(geopy.distance.geodesic((lat, lng), estimated_point).km)

  #res = torch.mean(torch.tensor(dist))#.to(device)
  #res.grad_fn = grad_fn
  return dist

tf_train = transforms.Compose([
  transforms.RandomResizedCrop((224, 224), (0.7, 1)),
  transforms.RandomHorizontalFlip(),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

tf_test = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def train_model():
    run = wandb.init(name="zone")
    print("Starting new run with configs:", wandb.config)

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    lr_decay = wandb.config.lr_decay
    zone_size = wandb.config.zone_size
    architecture = wandb.config.architecture
    data_normalization = wandb.config.data_normalization

    train = NSVD_Boxes('./data', transforms=tf_train, zone_size=zone_size, data_normalization=data_normalization, train=True)
    test = NSVD_Boxes('./data', transforms=tf_test, zone_size=zone_size, data_normalization=data_normalization, train=False)
    train_ldr = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4)
    test_ldr = DataLoader(test, batch_size=batch_size, shuffle=True, num_workers=4)

    model = get_model(architecture, 18 if zone_size == 'x200' else 7, False)
    model = model.to(device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)
    for epoch in range(epochs):
        print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
        model.train()
        epoch_loss = 0
        for batch_idx, (batch, labels) in (pbar := tqdm(enumerate(train_ldr), total=len(train_ldr))):
            batch, labels = batch.to(device), labels['box_index'].to(device)
            y = model(batch)
            loss = loss_fn(y, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            pbar.set_description(" loss: {:.5f}".format(epoch_loss/(batch_idx + 1)))
            
        with torch.no_grad():
            model.eval()
            test_loss = 0
            for batch_idx, (batch, labels) in (pbar := tqdm(enumerate(test_ldr), total=len(test_ldr))):
                batch, labels = batch.to(device), labels
                y = model(batch)
                avd_dist = distance_from_point(y, labels)
                test_loss += np.mean(avd_dist)
                pbar.set_description("test:  epoch: {:2d};  acc: {:.5f}".format(epoch, (test_loss/(batch_idx + 1))))

        wandb.log({
            "epoch": epoch,
            "epoch_lr": scheduler.get_last_lr()[0],
            "train_loss": epoch_loss/(batch_idx + 1),
            "test_loss": test_loss/(batch_idx + 1)
        })

        scheduler.step()
    run.finish()

if __name__ == '__main__':
    train_model()


