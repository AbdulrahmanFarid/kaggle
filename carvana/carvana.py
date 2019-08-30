import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from dataset import *
from model import *

transform_dataset = torchvision.transforms.ToTensor()
carvana2 = carvana_dataset('../input/', transform = transform_dataset)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = UNet(3, 1).to(device)
optimizer = optim.SGD(
    net.parameters(),
    lr = .1,
    momentum = .9,
    weight_decay=0.0005
)

#criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

for epoch in range(3):
    print('starting epoch number {}'.format(epoch))
    epoch_loss = 0
    net.train()
    for i, image_mask in enumerate(carvana2):
      #  try:
            original_mask = image_mask['mask'].to(device)
           # original_mask_flat = original_mask.squeeze().view(-1)
            #mask_pred = net(image_mask['image'].to(device).unsqueeze(0))
            mask_pred = net(image_mask['image'].unsqueeze(0).to(device))
           # print(mask_pred.shape, mask_pred.squeeze().shape)
           # mask_pred_flat= mask_pred.squeeze().view(-1)
         #   loss = criterion(original_mask_flat, mask_pred_flat)
            print(mask_pred.view(1, -1).shape)
            print(original_mask.view(1, -1).shape)
            loss = criterion( mask_pred.view(-1), original_mask.view(-1))
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    #    except RuntimeError as e:
#                 del original_mask
#                 del original_mask_flat
#                 del mask_pred
#                 del mask_pred_flat
#                 del loss
#                 print(e)


            if (i % 100) == 0:
                 print("train loss in {:0>2}epoch  /{:>5}iter:    {:<10.8}".\
                   format(epoch+1, i, epoch_loss/(i+1)))
            print("time number: %i", i)
