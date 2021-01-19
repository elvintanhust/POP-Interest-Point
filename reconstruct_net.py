import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# the architecture of reconstructor

class ReconstructNet(nn.Module):
    def __init__(self, feature_len, in_f_len):
        super(ReconstructNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 1
            nn.Conv2d(64, feature_len, kernel_size=1, padding=0)
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(feature_len, in_f_len * 8, 2, 1, 0, bias=False),
            nn.BatchNorm2d(in_f_len * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(in_f_len * 8, in_f_len * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_f_len * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(in_f_len * 4, in_f_len * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_f_len * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(in_f_len * 2, in_f_len, 4, 2, 1, bias=False),
            nn.BatchNorm2d(in_f_len),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.Conv2d(in_f_len, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 16 x 16
        )

    def forward(self, input):
        return self.decoder(self.encoder(input))
