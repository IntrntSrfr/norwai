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
    #self.conv5 = ConvModule(256, 256)
    self.l1 = nn.Linear(256*8*8, 1024)
    self.l2 = nn.Linear(1024, 2)
    
  def forward(self, x):
    x = self.pool(self.conv1(x)) # 128 -> 64
    x = self.pool(self.conv2(x)) # 64 -> 32
    x = self.pool(self.conv3(x)) # 32 -> 16
    x = self.pool(self.conv4(x)) # 16 -> 8
    #x = self.pool(self.conv5(x)) # 16 -> 8
    x = x.reshape(-1, 256*8*8)
    x = torch.relu(self.l1(x))
    x = self.l2(x)
    return x

if __name__ == '__main__':
  from nsvd import NSVD
  from tqdm import tqdm

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("using device:", device)

  train = NSVD()
  train_inputs = train.data.to(device)
  train_targets = train.targets.to(device)

  mean = train_inputs.mean()
  std = train_inputs.std()
  train_inputs = (train_inputs - mean) / std

  batch_size = 36
  train_input_batches = torch.split(train_inputs, batch_size)
  train_target_batches = torch.split(train_targets, batch_size)
  
  print("train input batch size:", train_input_batches[0].shape)
  print("train target batch size:", train_target_batches[0].shape)

  model = NSVDModel()
  model.to(device)

  loss_fn = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)

  epochs = 11
  for epoch in range(epochs):
    epoch_loss = 0
    for batch in (pbar := tqdm(range(len(train_input_batches)), ascii=True)):
      y = model(train_input_batches[batch])
      loss = loss_fn(y, train_target_batches[batch])
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      epoch_loss += loss
      pbar.set_description("epoch: {:2d}; loss: {:.3f}".format(epoch, epoch_loss/(batch+1)))
  torch.save(model, './data/model_regression')
