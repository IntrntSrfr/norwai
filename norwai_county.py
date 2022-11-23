import torch
from torch import nn 
from torchvision import models

class ConvModule(nn.Module):
  def __init__(self, in_features, out_features) -> None:
    super(ConvModule, self).__init__()
    self.conv = nn.Conv2d(in_features, out_features, 5, padding=2)
    self.norm = nn.BatchNorm2d(out_features)

  def forward(self, x) -> torch.Tensor:
    x = self.norm(self.conv(x))
    return torch.relu(x)

class NSVDModel(nn.Module):
  def __init__(self, num_classes) -> None:
    super(NSVDModel, self).__init__()
    self.num_classes = num_classes

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
      nn.Dropout(),
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(512, self.num_classes),
    )
  
  def forward(self, x: torch.Tensor):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

  def __len__(self):
    return sum(p.numel() for p in self.parameters())


if __name__ == '__main__':
  model = NSVDModel(10)
  print("params:", len(model))


def set_requires_grad(model: nn.Module, feature_extract: bool):
  if feature_extract:
    for param in model.parameters():
      param.requires_grad = False

def get_model(architecture: str, num_classes: int, feature_extract: bool, use_pretrained=True) -> nn.Module | None:
  model = None
  
  if architecture == "NSVD":
    return NSVDModel(num_classes)

  elif architecture == "ResNet50":
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(progress=False, weights=weights if use_pretrained else None)
    set_requires_grad(model, feature_extract)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

  elif architecture == "ResNet152":
    weights = models.ResNet152_Weights.DEFAULT
    model = models.resnet152(progress=False, weights=weights if use_pretrained else None)
    set_requires_grad(model, feature_extract)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

  elif architecture == "vgg19":
    weights = models.VGG19_BN_Weights.DEFAULT
    model = models.vgg19_bn(progress=False, weights=weights if use_pretrained else None)
    set_requires_grad(model, feature_extract)
    model.classifier[6] = nn.Linear(4096, num_classes)

  elif architecture == "EfficientNet":
    weights = models.EfficientNet_V2_S_Weights.DEFAULT
    model = models.efficientnet_v2_s(progress=False, weights=weights if use_pretrained else None)
    set_requires_grad(model, feature_extract)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

  return model
