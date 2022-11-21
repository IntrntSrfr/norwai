import wandb 
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from util import make_data_folders
from norwai_regression import NSVDModel, Haversine
from nsvd import NSVD4

make_data_folders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print("using device:", device)

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

train_data = NSVD4('./data', True,  "coords", False, transforms=tf_train)
test_data = NSVD4('./data', False,  "coords", False, transforms=tf_test)

def train_model():
  run = wandb.init(name='distance')
  print("new run with configs:", wandb.config)

  epochs = wandb.config.epochs
  lr = wandb.config.lr
  lr_decay = wandb.config.lr_decay
  batch_size = wandb.config.batch_size

  model = NSVDModel()
  model = nn.DataParallel(model)
  model.to(device)
  
  train_ldr = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
  test_ldr = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
  print("training data: {} images, {} batches".format(len(train_data), len(train_ldr)))
  print("test data: {} images, {} batches".format(len(test_data), len(test_ldr)))

  loss_fn = Haversine()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
  if lr_decay:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

  #losses = []
  for epoch in range(epochs):
    print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
    model.train()
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

    with torch.no_grad():
      model.eval()
      epoch_acc = 0
      for batch_idx, (batch, labels, _) in (pbar := tqdm(enumerate(test_ldr), total=len(test_ldr))):
        batch, labels = batch.to(device), labels.to(device)
        y = model(batch)
        loss = loss_fn(y[:, 0], y[:, 1], labels[:, 0], labels[:, 1])
        epoch_acc += loss.item()
        pbar.set_description("    acc: {:.5f}".format(epoch_acc / (batch_idx + 1)))

    wandb.log({
      'epoch':epoch,
      'train_loss': epoch_loss / len(train_ldr),
      'test_loss': epoch_acc / len(test_ldr)
    })
    
    scheduler.step()
  run.finish()

if __name__ == '__main__':
  train_model()


#wandb.agent(sweep_id, function=train_model)