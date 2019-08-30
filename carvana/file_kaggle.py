import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import os
import cv2
from PIL import Image
class carvana_dataset(Dataset):
    def __init__(self, path, train = 'train', mask = 'train_masks', transform = None):
        self.images = sorted([os.path.join(path, train, image) for image in os.listdir(os.path.join(path, train))])
        self.masks= sorted([os.path.join(path, mask, masking) for masking in os.listdir(os.path.join(path, mask))])
        self.transform = transform
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(self.images[idx])
        mask = np.asarray(Image.open(self.masks[idx]))

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return {'image': image, 'mask': mask}

import torch
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            #the original paper didn't use padding but here he used it(i don't know why)
            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv(in_ch, out_ch)
    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            #here he didn't use stride (i don't know why again)
            nn.MaxPool2d(2), #, stride = 2),
            double_conv(in_ch, out_ch)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners=True)
        else:
            #i don't know what he did in the next line exactly as it sound weired
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride = 2)
        self.conv = double_conv(in_ch, out_ch)
        #here x1 for previous layer
        #x2 for the corresponding layer(refer to paper)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #supposed he make copy and crop of x2 to get size of x1 not making padding
        #i don't know again why here he made padding but let's make as he did
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x1, x2], dim = 1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(UNet, self).__init__()
        self.inc = double_conv(in_ch, 64)
        self.down1 = down(64, 128)
#        print(self.down1)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.out = outconv(64, num_classes)
        
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)
        return torch.sigmoid(x)


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
