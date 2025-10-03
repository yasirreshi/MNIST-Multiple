# Import necessary libraries from PyTorch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # out: 26x26 | params: Conv 1*16*3*3=144, BN 2*16=32 => 176 | RF: 3

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) # out: 26x26 | params: Conv 16*12*3*3=1728, BN 24 => 1752 | RF: 5

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) # out: 26x26 | params: Conv 12*12*3*3=1296, BN 24 => 1320 | RF: 7

        self.pool1 = nn.MaxPool2d(2, 2) # out: 13x13 | params: 0 | RF: 8 (jump=2)

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # out: 11x11 | params: Conv 12*10*3*3=1080, BN 20 => 1100 | RF: 12

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) # out: 9x9 | params: Conv 10*10*3*3=900, BN 20 => 920 | RF: 16

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.1)
        ) # out: 7x7 | params: Conv 10*10*3*3=900, BN 20 => 920 | RF: 20

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # out: 5x5 | params: Conv 10*10*1*1=100, BN 20 => 120 | RF: 20 (1x1 conv doesn't grow RF)

        self.gap = nn.AvgPool2d(5) # out: 1x1 from 5x5 | params: 0 | RF: 28 (adds (5-1)*jump=8)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)             # GAP -> [batch, 10, 1, 1]
        x = x.view(x.size(0), -1)   # Flatten to [batch, 10]
        return F.log_softmax(x, dim=-1)