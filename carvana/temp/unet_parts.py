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
        return

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
        return xkk
