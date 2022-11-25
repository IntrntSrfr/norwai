import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from util import make_data_folders
from norwai_models import get_model
from nsvd import NSVD, NSVD_B, IDX2COUNTY

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

#train_data = NSVD_B('./data', True,  "county", False, transforms=tf_train)
#test_data = NSVD_B('./data', False,  "county", False, transforms=tf_test)
#
#default_config = {
#  'architecture': 'EfficientNet',
#  'pretrained': True,
#  'dataset': 'balanced',
#  'epochs': 10,
#  'lr': 0.001,
#  'lr_decay': 0.98,
#  'batch_size': 32,
#}

def get_dataset(full):
  if full:
    return NSVD('./data', True,  "county", False, transforms=tf_train), NSVD('./data', False,  "county", False, transforms=tf_train)
  else:
    return NSVD_B('./data', True,  "county", False, transforms=tf_train), NSVD_B('./data', False,  "county", False, transforms=tf_train)


def train_model():
  run = wandb.init(name='county')#, project='norwai', entity='norwai', config=default_config)
  print("new run with configs:", wandb.config)

  architecture = wandb.config.architecture
  pretrained = wandb.config.pretrained
  epochs = wandb.config.epochs
  lr = wandb.config.lr
  lr_decay = wandb.config.lr_decay
  batch_size = wandb.config.batch_size
  dataset = wandb.config.dataset

  train_data, test_data = get_dataset(dataset == 'full')

  model = get_model(architecture, 10, False, pretrained)
  assert model is not None, "Input architecture is invalid"
  model = nn.DataParallel(model)
  model.to(device)
  
  train_ldr = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
  test_ldr = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
  print("training data: {} images, {} batches".format(len(train_data), len(train_ldr)))
  print("test data: {} images, {} batches".format(len(test_data), len(test_ldr)))

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, 1)
  if lr_decay:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, lr_decay)

  for epoch in range(epochs):
    print("epoch: {}; lr: {}".format(epoch, scheduler.get_last_lr()[0]))
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (batch, labels, _) in (pbar := tqdm(enumerate(train_ldr), total=len(train_ldr))):
      batch, labels = batch.to(device), labels.to(device)
      y = model(batch)
      pred = torch.argmax(torch.exp(y), dim=1)
      loss = loss_fn(y, labels)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      train_loss += loss.item()
      train_acc += (pred == labels).sum().item() / len(batch)
      pbar.set_description("    loss: {:.3f}".format(train_loss / (batch_idx + 1)))
    
    truths = torch.tensor([]).to(device)
    predictions = torch.tensor([]).to(device)
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
      for batch_idx, (batch, labels, _) in (pbar := tqdm(enumerate(test_ldr), total=len(test_ldr))):
        batch, labels = batch.to(device), labels.to(device)
        y = model(batch)
        pred = torch.argmax(torch.exp(y), dim=1)
        loss = loss_fn(y, labels)
        if epoch == epochs - 1:
          truths = torch.cat((truths, labels))
          predictions = torch.cat((predictions, pred))
        test_loss += loss.item()
        test_acc += (pred == labels).sum().item() / len(batch)
        pbar.set_description("    acc: {:.3f}".format(test_acc / (batch_idx + 1)))
    
    if epoch == epochs - 1:
      cm = wandb.plot.confusion_matrix(
        y_true=truths.flatten().cpu().detach().numpy(),
        preds=predictions.flatten().cpu().detach().numpy(),
        class_names=list(IDX2COUNTY.values())
      )
      wandb.log({'conf_mat':cm})

    
    wandb.log({
      'epoch':epoch,
      'epoch_lr': scheduler.get_last_lr()[0],
      'train_loss': train_loss / len(train_ldr),
      'train_acc': train_acc / len(train_ldr),
      'test_loss': test_loss / len(test_ldr),
      'test_acc': test_acc / len(test_ldr)
    })

    scheduler.step()
  run.finish()

if __name__ == '__main__':
  train_model()