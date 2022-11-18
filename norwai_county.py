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
      ConvModule(64, 128),
      nn.MaxPool2d(2), # 512 -> 256
      ConvModule(128, 256),
      ConvModule(256, 512),
      nn.MaxPool2d(2), # 256 -> 128
      ConvModule(512, 512),
      ConvModule(512, 512),
      nn.MaxPool2d(2), # 128 -> 64
      ConvModule(512, 512),
      ConvModule(512, 512),
      nn.MaxPool2d(2), # 64 -> 32
      ConvModule(512, 512),
      ConvModule(512, 512),
      nn.MaxPool2d(2), # 32 -> 16
      ConvModule(512, 512),
      ConvModule(512, 512),
      nn.MaxPool2d(2), # 16 -> 8
    )
    
    self.classifier = nn.Sequential(
      nn.Linear(512*8*8, 1024),
      nn.ReLU(True),
      nn.Linear(1024, 1024),
      nn.ReLU(True),
      nn.Linear(1024, 10),
    )
    
  def forward(self, x: torch.Tensor):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from torchvision import transforms, models
  from nsvd import NSVD4
  from tqdm import tqdm
  import pandas as pd

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  tf = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  train = NSVD4('./data', 'county', False, transforms=tf)
  train_ldr = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
  test = NSVD4('./data', 'county', False, train=False, transforms=tf)
  test_ldr = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)

  print("training data: {} images, {} batches".format(len(train), len(train_ldr)))
  print("testing data: {} images, {} batches".format(len(test), len(test_ldr)))

  #model = NSVDModel()
  model = models.resnet152(progress=False, weights=models.ResNet152_Weights.IMAGENET1K_V1)
  model.fc = nn.Linear(512, 10)
  #for i, param in enumerate(model.features.parameters()):
  #  param.requires_grad = False
  #model.classifier[6] = nn.Linear(4096, 10)
  model.to(device)
  model_name = "resnet152_512_10"

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)

  losses = []
  accs = []

  epochs = 10
  for epoch in range(epochs):
    print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
    epoch_loss = 0
    for batch_idx, (batch, labels) in (pbar := tqdm(enumerate(train_ldr), total=len(train_ldr))):
      batch, labels = batch.to(device), labels.to(device)
      y = model(batch)
      loss = loss_fn(y, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss.item() / len(train_ldr)

      pbar.set_description("    loss: {:.5f}".format(epoch_loss))
    losses.append(epoch_loss)
    scheduler.step()

    with torch.no_grad():
      epoch_acc = 0
      for batch_idx, (batch, labels) in (pbar := tqdm(enumerate(test_ldr), total=len(test_ldr))):
        batch, labels = batch.to(device), labels.to(device)
        y = model(batch)
        y = torch.argmax(torch.exp(y), dim=1)
        epoch_acc += (1 - torch.count_nonzero(y - labels).item() / len(batch)) / len(test_ldr)
        pbar.set_description("    acc: {:.5f}".format(epoch_acc))
    accs.append(epoch_acc)

  torch.save(model, './data/trained_models/county/{}'.format(model_name))
  df = pd.DataFrame({'accuracy': accs, 'losses': losses})
  df.to_csv('./data/trained_models/county/metrics/{}.csv'.format(model_name), index=False)
