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
    self.pool = nn.MaxPool2d(2)
    self.conv1 = ConvModule(3, 64)
    self.conv2 = ConvModule(64, 128)
    self.conv3 = ConvModule(128, 256)
    self.conv4 = ConvModule(256, 256)
    self.l1 = nn.Linear(256*8*8, 512)
    self.l2 = nn.Linear(512, 11)
    
  def forward(self, x):
    x = self.pool(self.conv1(x)) # 128 -> 64
    x = self.pool(self.conv2(x)) # 64 -> 32
    x = self.pool(self.conv3(x)) # 32 -> 16
    x = self.pool(self.conv4(x)) # 16 -> 8
    x = x.reshape(-1, 256*8*8)
    x = torch.relu(self.l1(x))
    x = self.l2(x)
    return x

if __name__ == '__main__':
  from torch.utils.data import DataLoader
  from torchvision import transforms, datasets, models
  from nsvd import NSVD3
  from tqdm import tqdm, trange

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  tf = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
  ])

  train = NSVD3('./data', transforms=tf)
  train_ldr = DataLoader(train, batch_size=32, shuffle=True, num_workers=4)
  test = NSVD3('./data', train=False, transforms=tf)
  test_ldr = DataLoader(test, batch_size=32, shuffle=True, num_workers=4)

  #model = NSVDModel()
  model = models.vgg11_bn(progress=False, weights=models.VGG11_BN_Weights.IMAGENET1K_V1)
  for i, param in enumerate(model.features.parameters()):
    param.requires_grad = False
  model.classifier[6] = nn.Linear(4096, 11)
  model.to(device)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
  #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.87)

  losses = []
  accs = []

  epochs = 5
  for epoch in range(epochs):
    epoch_loss = 0
    pbar = trange(len(train_ldr), ascii=True)
    for batch_idx, (batch, labels) in enumerate(train_ldr):
      batch, labels = batch.to(device), labels.to(device)
      y = model(batch)
      loss = loss_fn(y, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss.item()/len(train_ldr)

      pbar.set_description("train: epoch: {:2d}; loss: {:.5f}".format(epoch, epoch_loss))
      pbar.update()
    losses.append(epoch_loss)
    pbar.close()
    #scheduler.step()

    pbar = trange(len(test_ldr), ascii=True)
    with torch.no_grad():
      epoch_acc = 0
      for batch_idx, (batch, labels) in enumerate(test_ldr):
        batch, labels = batch.to(device), labels.to(device)
        y = model(batch)
        y = torch.argmax(torch.exp(y), dim=1)
        epoch_acc += (1-torch.count_nonzero(y-labels).item() / len(batch))/len(test_ldr)
        pbar.set_description("test:  epoch: {:2d};  acc: {:.5f}".format(epoch, epoch_acc))
        pbar.update()
    accs.append(epoch_acc)
    pbar.close()
  torch.save(model, './data/model_classification')

  import matplotlib.pyplot as plt  
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
  fig.set_facecolor('white')
  ax1.set_yscale('log')
  ax1.plot(range(len(losses)), losses, label='loss')
  ax1.legend()
  ax2.plot(range(len(accs)), accs, label='accuracy')
  ax2.legend()
  plt.show()