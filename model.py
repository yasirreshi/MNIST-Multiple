# Import necessary libraries from PyTorch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # 26x26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # 26x26

        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        ) # 26x26

        self.pool1 = nn.MaxPool2d(2, 2) # 13x13

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=16, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # 11x11

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=14, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # 9x9

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=12, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # 7x7

        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=10, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # 5x5

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=0, bias=False),
            nn.ReLU()
        ) # 3x3

        self.fc = nn.Linear(10*3*3, 10)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = x.view(x.size(0), -1) # Flatten to [batch, 10*3*3]
        x = self.fc(x)             # [batch, 10]
        return F.log_softmax(x, dim=-1)