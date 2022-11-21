import torch
from torch import nn

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
      nn.MaxPool2d(2), # 224 -> 112
      ConvModule(64, 128),
      nn.MaxPool2d(2), # 112 -> 56
      ConvModule(128, 256),
      nn.MaxPool2d(2), # 56 -> 28
      ConvModule(256, 256),
      nn.MaxPool2d(2), # 28 -> 14
      ConvModule(256, 256),
      nn.MaxPool2d(2), # 14 -> 7
    )
    
    self.classifier = nn.Sequential(
      nn.Linear(256*7*7, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Linear(512, 2),
    )
  
  def forward(self, x: torch.Tensor):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

class Haversine(nn.Module):
  def __init__(self) -> None:
    super(Haversine, self).__init__()
  
  def forward(self, lat1: torch.Tensor, lng1: torch.Tensor, lat2: torch.Tensor, lng2: torch.Tensor) -> torch.Tensor:
    lng1, lat1, lng2, lat2 = map(torch.deg2rad, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1
    dlat = lat2 - lat1
    a = torch.sin(dlat/2.0)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlng/2.0)**2
    c = 2 * torch.arcsin(torch.sqrt(a))
    km = 6371 * c
    return km.mean()
