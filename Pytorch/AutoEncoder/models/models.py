import torch
import torch.nn as nn
import torch.nn.functional as F

class LittmanNet(nn.Module):

    def __init__(self):
        super(LittmanNet, self).__init__()

        # Method for up sampling image.
        self.up_sample_mode = 'bicubic'

        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # === Encoder ===
        # convolve 3x32x32 --> 4x32x32
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        # down sample 4x32x32 --> 4x16x16
        self.down_pool1 = nn.MaxPool2d(2, 2)
        # convolve 4x16x16 --> 5x16x16
        self.conv2 = nn.Conv2d(4, 5, kernel_size=3, padding=1)
        # down sample 5x16x16 --> 5x8x8
        self.down_pool2 = nn.MaxPool2d(2, 2)
        # flatten 5x8x8 --> 1x320
        # dense 1x320 --> 1x80
        self.fc1 = nn.Linear(5*8*8, 80)
        # dense 1x80 --> 1x24
        self.fc2 = nn.Linear(80, 24)

        # === Decoder ===
        # dense 1x24 --> 1x80
        self.fc3 = nn.Linear(24, 80)
        # dense 1x80 --> 1x320
        self.fc4 = nn.Linear(80, 320)
        # reshape 1x320 --> 5x8x8
        # up sample 5x8x8 --> 5x16x16
        self.up_pool1 = nn.Upsample(scale_factor=2, mode=self.up_sample_mode)
        # convolve 5x16x16 --> 4x16x16
        self.conv3 = nn.Conv2d(5, 4, kernel_size=3, padding=1)
        # up sample 4x16x16 --> 4x32x32
        self.up_pool2 = nn.Upsample(scale_factor=2, mode=self.up_sample_mode)
        # convolve 4x32x32 --> 3x32x32
        self.conv4 = nn.Conv2d(4, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Encode unactivated x using encoder part of network.
        # Returns unactivated code.
        x = self.down_pool1(F.relu(self.conv1(x)))
        x = self.down_pool2(F.relu(self.conv2(x)))
        # reshape
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def decode(self, x):
        # Decode x using decoder part of network.
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # reshape
        x = x.view(-1, 5, 8, 8)
        x = self.up_pool1(F.relu(self.conv3(x)))
        x = self.up_pool2(F.relu(self.conv4(x)))
        return x

