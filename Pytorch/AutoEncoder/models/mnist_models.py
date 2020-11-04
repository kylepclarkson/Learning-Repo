# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:32:04 2020

@author: Kyle
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


# =====================
# === Large Network ===
# =====================
class NetLarge(nn.Module):
    
    def __init__(self, name='NetLarge', ckp_dir='saved_models/'):
        super(NetLarge, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 1x28x28 --> 64x28x28        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # downsample 64x28x28 --> 64x14x14
        self.downsample1 = nn.MaxPool2d(2, 2)
        # convolve 64x14x14 --> 32x14x14
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # downsample 32x14x14 --> 12x7x7
        self.downsample2 = nn.MaxPool2d(2, 2)
        # convolve 12x7x7 --> 12x7x7
        self.conv3 = nn.Conv2d(32, 12, kernel_size=3, padding=1)
        # flatten 12x7x7 --> 1x588
        # dense 1x588 --> 1x24
        self.fc1 = nn.Linear(12*7*7, 24)

        # === Decoder ===
        # dense 1x24 --> 1x588
        self.fc2 = nn.Linear(24, 12*7*7)
        # reshape 1x588 --> 12x7x7
        # deconvolve 12x7x7 --> 32x7x7
        self.deconv1 = nn.ConvTranspose2d(12, 32, kernel_size=1, stride=1)
        # deconvolve 32x7x7 --> 64x14x14
        self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        # deconvolve 64x14x14 --> 1x28x28
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(F.relu(x))
        return x
    
    def encode(self, x):
        
        x = self.downsample1(F.relu(self.conv1(x)))
        x = self.downsample2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # reshape
        x = x.view(-1, 12*7*7)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.fc2(x))
        # reshape
        x = x.view(-1, 12, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
                
    def summary(self):
        self.to(self.device)
        return summary(self, (1, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        self.name = name
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, self.name+'_mnist.pt')
        
        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


# ==============================
# === Large Network, Dropout ===
# ==============================
class NetLargeDropout(nn.Module):
    
    def __init__(self, name='NetLargeDropout', ckp_dir='saved_models/'):
        super(NetLargeDropout, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 1x28x28 --> 64x28x28        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        # downsample 64x28x28 --> 64x14x14
        self.downsample1 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout1 = nn.Dropout(0.3)
        # convolve 64x14x14 --> 32x14x14
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # downsample 32x14x14 --> 32x7x7
        self.downsample2 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout2 = nn.Dropout(0.3)
        # convolve 32x7x7 --> 12x7x7
        self.conv3 = nn.Conv2d(32, 12, kernel_size=3, padding=1)
        # flatten 12x7x7 --> 1x588
        # dense 1x588 --> 1x24
        self.fc1 = nn.Linear(12*7*7, 24)

        # === Decoder ===
        # dense 1x24 --> 1x588
        self.fc2 = nn.Linear(24, 12*7*7)
        # reshape 1x588 --> 12x7x7
        # deconvolve 12x7x7 --> 32x7x7
        self.deconv1 = nn.ConvTranspose2d(12, 32, kernel_size=1, stride=1)
        # deconvolve 32x7x7 --> 64x14x14
        self.deconv2 = nn.ConvTranspose2d(32, 64, kernel_size=2, stride=2)
        # deconvolve 64x14x14 --> 1x28x28
        self.deconv3 = nn.ConvTranspose2d(64, 1, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(F.relu(x))
        return x
    
    def encode(self, x):
        
        x = self.dropout1(self.downsample1(F.relu(self.conv1(x))))
        x = self.dropout2(self.downsample2(F.relu(self.conv2(x))))
        x = F.relu(self.conv3(x))
        # reshape
        x = x.view(-1, 12*7*7)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.fc2(x))
        # reshape
        x = x.view(-1, 12, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
                
    def summary(self):
        self.to(self.device)
        return summary(self, (1, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        self.name = name
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, self.name+'_mnist.pt')
        
        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

# =====================
# === Small Network ===
# =====================
class NetSmall(nn.Module):
    
    def __init__(self, name='NetSmall', ckp_dir='saved_models/'):
        super(NetSmall, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 1x28x28 --> 6x28x28        
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, padding=1)
        # downsample 6x28x28 --> 6x14x14
        self.downsample1 = nn.MaxPool2d(2, 2)
        # convolve 6x14x14 --> 12x14x14
        self.conv2 = nn.Conv2d(6, 12, kernel_size=3, padding=1)
        # downsample 12x14x14 --> 12x7x7
        self.downsample2 = nn.MaxPool2d(2, 2)
        # convolve 12x7x7 --> 12x7x7
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        # flatten 12x7x7 --> 1x588
        # dense 1x588 --> 1x24
        self.fc1 = nn.Linear(12*7*7, 24)

        # === Decoder ===
        # dense 1x24 --> 1x588
        self.fc2 = nn.Linear(24, 12*7*7)
        # reshape 1x588 --> 12x7x7
        # deconvolve 12x7x7 --> 12x7x7
        self.deconv1 = nn.ConvTranspose2d(12, 12, kernel_size=1, stride=1)
        # deconvolve 12x7x7 --> 6x14x14
        self.deconv2 = nn.ConvTranspose2d(12, 6, kernel_size=2, stride=2)
        # deconvolve 6x14x14 --> 1x28x28
        self.deconv3 = nn.ConvTranspose2d(6, 1, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(F.relu(x))
        return x
    
    def encode(self, x):
        
        x = self.downsample1(F.relu(self.conv1(x)))
        x = self.downsample2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # reshape
        x = x.view(-1, 12*7*7)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.fc2(x))
        # reshape
        x = x.view(-1, 12, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
                
    def summary(self):
        self.to(self.device)
        return summary(self, (1, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        self.name = name
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, self.name+'_mnist.pt')
        
        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

# ========================
# === Testing Networks ===         
# ========================

class Clapton(nn.Module):
    
    def __init__(self, name='Clapton', ckp_dir='saved_models/'):
        super(Clapton, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 1x28x28 --> 32x28x28        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # downsample 32x28x28 --> 32x14x14
        self.downsample1 = nn.MaxPool2d(2, 2)
        # convolve 32x14x14 --> 32x14x14
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # downsample 32x14x14 --> 32x7x7
        self.downsample2 = nn.MaxPool2d(2, 2)
        # convolve 32x7x7 --> 32x7x7
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        # flatten 32x7x7 --> 1x1568
        # dense 1x392 --> 1x24
        self.fc1 = nn.Linear(32*7*7, 24)
        
        # === Decoder ===
        # dense 1x24 --> 1x1568
        self.fc2 = nn.Linear(24, 32*7*7)
        # reshape 1x1568 --> 32x7x7
        # deconvolve 32x7x7 --> 32x7x7
        self.deconv1 = nn.ConvTranspose2d(32, 32, kernel_size=1, stride=1)
        # deconvolve 32x7x7 --> 32x14x14
        self.deconv2 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        # deconvolve 32x14x14 --> 1x28x28
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(F.relu(x))
        return x
    
    def encode(self, x):
        
        x = self.downsample1(F.relu(self.conv1(x)))
        x = self.downsample2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # reshape
        x = x.view(-1, 32*7*7)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.fc2(x))
        # reshape
        x = x.view(-1, 32, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
                
    def summary(self):
        self.to(self.device)
        return summary(self, (1, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        self.name = name
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, self.name+'_mnist.pt')
        
        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
# ======== End Clapton ========

class ChanSmall(nn.Module):
    
    def __init__(self, name='ChanSmall', ckp_dir='saved_models/'):
        super(ChanSmall, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 1x28x28 --> 4x28x28        
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # downsample 4x28x28 --> 4x14x14
        self.downsample1 = nn.MaxPool2d(2, 2)
        # convolve 4x14x14 --> 8x14x14
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # downsample 8x14x14 --> 8x7x7
        self.downsample2 = nn.MaxPool2d(2, 2)
        # convolve 8x7x7 --> 8x7x7
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        # flatten 64x7x7 --> 1x392
        # dense 1x392 --> 1x2
        self.fc1 = nn.Linear(8*7*7, 2)
        
        # === Decoder ===
        # dense 1x2 --> 1x392
        self.fc2 = nn.Linear(2, 8*7*7)
        # reshape 1x392 --> 8x7x7
        # deconvolve 8x7x7 --> 8x7x7
        self.deconv1 = nn.ConvTranspose2d(8, 8, kernel_size=1, stride=1)
        # deconvolve 8x7x7 --> 4x14x14
        self.deconv2 = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)
        # deconvolve 4x14x14 --> 1x28x28
        self.deconv3 = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2)
        
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(F.relu(x))
        return x
    
    def encode(self, x):
        
        x = self.downsample1(F.relu(self.conv1(x)))
        x = self.downsample2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # reshape
        x = x.view(-1, 8*7*7)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        x = F.relu(self.fc2(x))
        # reshape
        x = x.view(-1, 8, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
                
    def summary(self):
        self.to(self.device)
        return summary(self, (1, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        self.name = name
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, self.name+'_mnist.pt')
        
        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

# === End ChanSmall ===    

class Chan(nn.Module):
    
    def __init__(self, name='Chan', ckp_dir='saved_models/'):
        super(Chan, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 1x28x28 --> 32x28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # downsample 32x28x28 --> 32x14x14
        self.downsample1 = nn.MaxPool2d(2, 2)
        # convolve 32x14x14 --> 64x14x14
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # downsample 64x14x14 --> 64x7x7
        self.downsample2 = nn.MaxPool2d(2, 2)
        # convolve 64x7x7 --> 64x7x7
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        # flatten 64x7x7 --> 1x3136
        # dense 1x3136 --> 1x2
        self.fc1 = nn.Linear(64*7*7, 2)
        
        # === Decoder ===
        # dense 1x2 --> 1x 3136
        self.fc2 = nn.Linear(2, 64*7*7)
        # reshape 1x3136 --> 64x7x7
        # deconvolve 64x7x7 --> 64x7x7
        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=1, stride=1)
        # deconvolve 64x7x7 --> 32x14x14
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # deconvolve 32x14x14 --> 1x28x28
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)
        
        
    def forward(self, x):
        x = self.encode(x)
        x = self.decode(F.relu(x))
        return x
    
    def encode(self, x):
        
        x = self.downsample1(F.relu(self.conv1(x)))
        x = self.downsample2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # reshape
        x = x.view(-1, 64*7*7)
        x = self.fc1(x)
        return x
    
    def decode(self, x):
        
        x = F.relu(self.fc2(x))
        # reshape
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = self.deconv3(x)
        return x
                
    def summary(self):
        self.to(self.device)
        return summary(self, (1, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        self.name = name
        self.ckp_dir = ckp_dir
        self.ckp_file = os.path.join(self.ckp_dir, self.name+'_mnist.pt')
        
        # Set device for model.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

def save_checkpoint(model):
    print(f'=== Saving model {model.name} checkpoint === ')
    torch.save(model.state_dict(), model.ckp_file)

def load_checkpoint(model):
    print(f'=== Loading model {model.name} checkpoint ===')
    return torch.load(model.ckp_file)

    