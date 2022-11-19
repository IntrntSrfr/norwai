import torch
from torch import nn
import numpy as np

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

    self.features = nn.Sequential(
      ConvModule(3, 64),
      nn.MaxPool2d(2), # 128 -> 64
      ConvModule(64, 128),
      nn.MaxPool2d(2), # 64 -> 32
      ConvModule(128, 256),
      nn.MaxPool2d(2), # 32 -> 16
      ConvModule(256, 256),
      nn.MaxPool2d(2), # 16 -> 8
    )
    
    self.classifier = nn.Sequential(
      #nn.Linear(512*8*8, 1024),
      #nn.ReLU(True),
      nn.Linear(256*8*8, 512),
      #nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 2),
    )
  
  def forward(self, x: torch.Tensor):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


def haversine(lat1: torch.Tensor, lng1: torch.Tensor, lat2: torch.Tensor, lng2: torch.Tensor) -> torch.Tensor:
    lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = torch.sin(dlat/2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng/2.0)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = 6371 * c
    return km.mean()


def train_model(model, dataset, epochs, lr, batch_size):
  train_ldr = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
  print("training data: {} images, {} batches".format(len(dataset), len(train_ldr)))


  loss_fn = haversine
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

  losses = []
  for epoch in range(epochs):
    print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
    epoch_loss = 0
    for batch_idx, (batch, labels, _) in (pbar := tqdm(enumerate(train_ldr), total=len(train_ldr))):
      batch, labels = batch.to(device), labels.to(device).float()
      y = model(batch)
      loss = loss_fn(y[:, 0], y[:, 1], labels[:, 0], labels[:, 1])
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss.item()
      pbar.set_description("    loss: {:.5f}".format(epoch_loss / (batch_idx + 1)))
    losses.append(epoch_loss / len(train_ldr))
    #writer.add_scalar('Loss/train', epoch_loss, epoch)
    #scheduler.step()

  return model, losses


if __name__ == '__main__':
  from itertools import product
  from torch.utils.tensorboard.writer import SummaryWriter
  from torch.utils.data import DataLoader
  from torchvision import transforms, models
  from nsvd import NSVD4
  from tqdm import tqdm
  import pandas as pd

  from util import make_data_folders
  make_data_folders()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  train_data = NSVD4('./data', "coords", False, transforms=tf)
  train_ldr = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=4)
  print("training data: {} images, {} batches".format(len(train_data), len(train_ldr)))

  #model = NSVDModel()  
  #model = models.resnet152(progress=False, weights=models.ResNet152_Weights.IMAGENET1K_V1)
  #model.fc = nn.Linear(512, 2)
  #for i, param in enumerate(model.features.parameters()):
  #  param.requires_grad = False
  #model.classifier[6] = nn.Linear(4096, 10)
  #model = nn.DataParallel(model)
  #model.to(device)
  #model_name = "resnet152_512_10"
  #model_name = "nsvd_128_5"

  
  hparams = {
    'lr': [0.00001, 0.0001],
    'batch_size': [32, 64, 128],
  }
  writer = SummaryWriter('./data/trained_models/distance/metrics')

  for lr, batch_size in product(*hparams.values()):
    model = NSVDModel()  
    model = nn.DataParallel(model)
    model.to(device)
    model_desc = "nsvd_128_{}_{}_{}".format(5, lr, batch_size)

    model, losses = train_model(model, train_data, 5, lr, batch_size)

    torch.save(model, './data/trained_models/distance/{}'.format(model_desc))
  
    for i, v in enumerate(losses):
      writer.add_scalar('Loss', v, i)

    writer.add_hparams(
      {'epochs': 5, 'lr': lr, 'img_res': 128, 'weight_decay': 0.0001, 'batch_size': batch_size}, 
      {'loss': losses[-1]}
    )
    writer.flush()
    
    df = pd.DataFrame({'losses': losses})
    df.to_csv('./data/trained_models/distance/metrics/{}.csv'.format(model_desc), index=False)

  writer.close()

