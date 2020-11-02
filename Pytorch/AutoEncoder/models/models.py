import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary

class Sreeni(nn.Module):
    
    def __init__(self, name='Sreeni', checkpoint_dir='saved_mdoels/'):
        super(Sreeni, self).__init__()

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_cifar10.pt')

        # Method for up sampling image.
        self.up_sample_mode = 'bicubic'

        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # === Encoder ===
        # convolve 3x32x32 --> 6x32x32
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
        # downsample 6x32x32 --> 6x16x16
        self.down_pool1 = nn.MaxPool2d(2, 2)
        # convolve 6x16x16 --> 8x16x16
        self.conv2 = nn.Conv2d(6, 8, kernel_size=3, padding=1)
        # downsample 8x16x16 --> 8x8x8
        self.down_pool2 = nn.MaxPool2d(2, 2)
        # flatten 8x8x8 --> 1x512
        # dense 1x512 --> 1x128
        self.fc1 = nn.Linear(512, 128)
        # dense 1x128 --> 1x24
        self.fc2 = nn.Linear(128, 24)
        
        # === Decoder ===
        # dense 1x24 --> 1x128
        self.fc3 = nn.Linear(24, 128)
        # dense 1x128 --> 1x512
        self.fc4 = nn.Linear(128, 512)        
        # reshape 1x512 --> 8x8x8
        # convolveT 8x8x8 --> 6x16x16
        self.convT1 = nn.ConvTranspose2d(8, 6, kernel_size=3, padding=2)
        # convolveT 6x16x16 --> 3x32x32
        self.convT2 = nn.ConvTranspose2d(6, 3, kernel_size=3, padding=2)
        
    def encode(self, x):
        # Encode x. Returns unactivated code. 
        x = self.down_pool1(F.relu(self.conv1(x)))
        x = self.down_pool2(F.relu(self.conv2(x)))
        # reshape
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))
        x = self.fc(2)
        return x
    
    def decode(self, x):
        # Decode x.
        x = self.fcT1()
        
        
    def summary(self):
        self.to(self.device)
        return summary(self, (3, 32, 32))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)

class LittmanNet(nn.Module):

    def __init__(self, name='Littman', checkpoint_dir='saved_models/'):
        super(LittmanNet, self).__init__()

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_cifar10.pt')

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
        # flatten 5x8x8 --> 1x80
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

    def summary(self):
        self.to(self.device)
        return summary(self, (3, 32, 32))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)


def save_checkpoint(model):
    print(f'=== Saving model {model.name} checkpoint === ')
    torch.save(model.state_dict(), model.checkpoint_file)

def load_checkpoint(model):
    print(f'=== Loading model {model.name} checkpoint ===')
    return torch.load(model.checkpoint_file)

