# Notebook parameters

BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 10
EARLY_STOPPING = 25
OUTPUT_DIR = 'outputs/output_VIT/'
output_dir = OUTPUT_DIR

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary

from sklearn.metrics import accuracy_score
from PIL import Image, ImageOps, ImageEnhance
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, transforms
from torchvision.models.resnet import resnet18
from sklearn.metrics import confusion_matrix
from einops import rearrange


#original  data
all_transforms = transforms.Compose([transforms.Resize(28),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)), ])

#augmentation data
#all_transforms = transforms.Compose([transforms.Resize(28),transforms.RandomHorizontalFlip(),transforms.RandomAffine(degrees=0, translate=(.05, .05), scale=None),transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)), ])

# Get train and test data
train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,transform=all_transforms)
test_data = datasets.FashionMNIST('../fashion_data', train=False,transform=all_transforms)

# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

import random
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    with open('results/VIT_wo_aug.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
            dim=64, depth=6, heads=8, mlp_dim=128).to(device)
summary(model, (1, 28, 28))
    
# # # get optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(params=model.parameters(),lr=1e-3,momentum=0.9,weight_decay=1e-5)

# # # get scheduler

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# # # get loss
loss_func = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    model = model.cuda()
    loss_func = loss_func.cuda()

best_val_accuracy = 0
min_val_loss = np.inf
best_epoch = 0
batches = 0
epochs_no_improve = 0
n_epochs_stop = EARLY_STOPPING
train_loss_data = []
val_loss_data = []
train_accuracy_data = []
val_accuracy_data = []
predictions_list = []
labels_list = []


for epoch in range(EPOCHS):
    running_loss = 0.0
    targets = torch.empty(size=(BATCH_SIZE, )).to(device) 
    outputs = torch.empty(size=(BATCH_SIZE, )).to(device)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batches += 1
        data, target = Variable(data), Variable(target)
        if torch.cuda.is_available():
            data = data.type(torch.FloatTensor).cuda()
            target = target.cuda()
        targets = torch.cat((targets, target), 0)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        output = torch.argmax(torch.softmax(output, dim=1), dim=1).to(device)
        outputs = torch.cat((outputs, output), 0)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('train/loss on EPOCH {}: {}'.format(epoch, running_loss/batches))
    train_acc = accuracy_score(targets.cpu().detach().numpy().astype(int), 
                              outputs.cpu().detach().numpy().astype(int))
    print('train/accuracy: {} for epoch {}'.format(train_acc, epoch))

    train_loss = running_loss/batches

    # Save train loss and accuracy
    train_loss_data.append(train_loss)
    train_accuracy_data.append(train_acc)



    model.eval()

    # Validation loop
    running_loss = 0.0
    batches = 0
    targets = torch.empty(size=(BATCH_SIZE, )).to(device) 
    outputs = torch.empty(size=(BATCH_SIZE, )).to(device) 
    for batch_idx, (data, target) in enumerate(test_loader):
        batches += 1
        data, target = Variable(data, volatile=True), Variable(target)
        if torch.cuda.is_available():
            data = data.type(torch.FloatTensor).cuda()
            target = target.cuda()
            labels_list.append(target)
        with torch.no_grad():
            targets = torch.cat((targets, target), 0)
            output = model(data)
            loss = loss_func(output, target)
            output = torch.argmax(torch.softmax(output, dim=1), dim=1).to(device)
            predictions_list.append(output)
            outputs = torch.cat((outputs, output), 0)
            running_loss += loss.item()

    val_loss = running_loss/batches
    print('val/loss: {}'.format(val_loss))
    val_acc = accuracy_score(targets.cpu().detach().numpy().astype(int), 
                      outputs.cpu().detach().numpy().astype(int))
    print('val/accuracy: {} for epoch {}'.format(val_acc, epoch))

    #Save validation loss and accuracy
    val_loss_data.append(val_loss)
    val_accuracy_data.append(val_acc)
   
   # Model Checkpoint for best validation f1
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        min_val_loss = val_loss
        print('Best val/acc: {} for epoch {}, saving model---->'.format(val_acc, epoch))
        torch.save(model.state_dict(), "{}/snapshot_epoch_{}.pth".format(output_dir, epoch))
        best_epoch = epoch
        epochs_no_improve = 0
    #else:
        #epochs_no_improve += 1
    #if epochs_no_improve == n_epochs_stop:
        #print('Early stopping!')
        #break
from itertools import chain 

predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
predictions_l = list(chain.from_iterable(predictions_l))
labels_l = list(chain.from_iterable(labels_l))

import sklearn.metrics as metrics

confusion_matrix(labels_l, predictions_l)
report = metrics.classification_report(labels_l, predictions_l)
classifaction_report_csv(report)
print("Classification report for CNN :\n%s\n"
      % (metrics.classification_report(labels_l, predictions_l)))

np.savetxt("loss/VIT_train.txt", train_loss_data, delimiter=',')
np.savetxt("loss/VIT_test.txt", val_loss_data, delimiter=',')
np.savetxt("accuracy/VIT_train.txt", train_accuracy_data, delimiter=',')
np.savetxt("accuracy/VIT_test.txt", val_accuracy_data, delimiter=',')



