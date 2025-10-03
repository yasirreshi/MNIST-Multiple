# Import necessary libraries from PyTorch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input: 28x28x1 | RF: 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Output: 28x28x10
        # RF: 1 + (3-1)*1 = 3
        # Params: (1*10*3*3) + (10*2) = 90 + 20 = 110

        self.convblock2 = nn.Sequential(
            nn.Conv2d(10, 14, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(14),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Output: 28x28x14
        # RF: 3 + (3-1)*1 = 5   
        # Params: (10*14*3*3) + (14*2) = 1260 + 28 = 1288

        self.pool1 = nn.MaxPool2d(2, 2)
        # Output: 14x14x14
        # RF: 5 + (2-1)*1 = 6
        # Params: 0 
        # Jump after pool: 1*2 = 2

        self.convblock3 = nn.Sequential(
            nn.Conv2d(14, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Output: 14x14x16
        # RF: 6 + (3-1)*2 = 10
        # Params: (14*16*3*3) + (16*2) = 2016 + 32 = 2048

        self.convblock4 = nn.Sequential(
            nn.Conv2d(16, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Output: 14x14x12
        # RF: 10 + (3-1)*2 = 14
        # Params: (16*12*3*3) + (12*2) = 1728 + 24 = 1752

        self.pool2 = nn.MaxPool2d(2, 2)
        # Output: 7x7x12
        # RF: 14 + (2-1)*2 = 16
        # Params: 0
        # Jump after pool: 1*2 = 2

        self.convblock5 = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Output: 7x7x12
        # RF: 16 + (3-1)*4 = 24
        # Params: (12*12*3*3) + (12*2) = 1296 + 24 = 1320

        self.convblock6 = nn.Sequential(
            nn.Conv2d(12, 10, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout(0.05),
        )
        # Output: 7x7x10
        # RF: 24 + (3-1)*4 = 32
        # Params: (12*10*3*3) + (10*2) = 1200 + 20 = 1220

        self.convblock7 = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=1, bias=False),
        )
        # Output: 7x7x10
        # RF: 32 + (1-1)*4 = 32
        # Params: (10*10*1*1) + (10*2) = 100 + 20 = 120

        self.gap = nn.AvgPool2d(7)
        # Output: 1x1x10
        # RF: The GAP layer's effective RF covers the entire feature map.
        # Here, it aggregates the 7x7 map, where each point has an RF of 32.
        # Params: 0

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.pool2(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return F.log_softmax(x, dim=-1)