# Notebook parameters

BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 10
EARLY_STOPPING = 25
OUTPUT_DIR = 'outputs/output_Mobilenet/'
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
from torchvision.models.mobilenet import MobileNetV2
from sklearn.metrics import confusion_matrix

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
    with open('results/Mobilenet_wo_aug.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')

model = MobileNetV2(num_classes=10)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
model = model.to(device)
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

np.savetxt("loss/Mobilenet_train.txt", train_loss_data, delimiter=',')
np.savetxt("loss/Mobilenet_test.txt", val_loss_data, delimiter=',')
np.savetxt("accuracy/Mobilenet_train.txt", train_accuracy_data, delimiter=',')
np.savetxt("accuracy/Mobilenet_test.txt", val_accuracy_data, delimiter=',')



