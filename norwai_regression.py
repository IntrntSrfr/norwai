import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from util import make_data_folders
from norwai_models import get_model, Haversine
from nsvd import NSVD, NSVD_B

make_data_folders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Variables
architecture = "NSVD" 
pretrained = True
epochs = 10
lr = 0.001
lr_decay = 0.95
batch_size = 32
dataset = 'full' # 'full' or 'balanced'

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

def get_dataset(full):
  if full:
    return NSVD('./data', True,  "coords", False, transforms=tf_train), NSVD('./data', False,  "coords", False, transforms=tf_train)
  else:
    return NSVD_B('./data', True,  "coords", False, transforms=tf_train), NSVD_B('./data', False,  "coords", False, transforms=tf_train)

train_data, test_data = get_dataset(dataset == 'full')

model = get_model(architecture, 2, False, pretrained)
assert model is not None, "Input architecture is invalid"
model = nn.DataParallel(model)
model.to(device)

train_ldr = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
test_ldr = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)

loss_fn = Haversine()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
if lr_decay:
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

for epoch in range(epochs):
  print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
  model.train()
  train_loss = 0
  for batch_idx, (batch, labels, _) in (pbar := tqdm(enumerate(train_ldr), total=len(train_ldr))):
    batch, labels = batch.to(device), labels.to(device).float()
    y = model(batch)
    loss = loss_fn(y[:, 0], y[:, 1], labels[:, 0], labels[:, 1])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    train_loss += loss.item()
    pbar.set_description("    loss: {:.2f}".format(train_loss / (batch_idx + 1)))

  with torch.no_grad():
    model.eval()
    test_loss = 0
    for batch_idx, (batch, labels, _) in (pbar := tqdm(enumerate(test_ldr), total=len(test_ldr))):
      batch, labels = batch.to(device), labels.to(device)
      y = model(batch)
      loss = loss_fn(y[:, 0], y[:, 1], labels[:, 0], labels[:, 1])
      test_loss += loss.item()
      pbar.set_description("    acc: {:.2f}".format(test_loss / (batch_idx + 1)))
  scheduler.step()

#Save the model
torch.save(model, './data/trained_models/norwai_regression.model')