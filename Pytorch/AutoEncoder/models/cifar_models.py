import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


# =====================
# === Large Network ===
# =====================
class NetLarge(nn.Module):
    
    def __init__(self, name='NetLarge', ckp_dir='saved_models/cifar10/'):
        super(NetLarge, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 3x28x28 --> 64x28x28        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
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
        # deconvolve 64x14x14 --> 3x28x28
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        
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
        return summary(self, (3, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        set_model_name(self, name, ckp_dir)


# ==============================
# === Large Network, Dropout ===
# ==============================
class NetLargeDropout(nn.Module):
    
    def __init__(self, name='NetLargeDropout', ckp_dir='saved_models/cifar10/'):
        super(NetLargeDropout, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 3x28x28 --> 64x28x28        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
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
        # deconvolve 64x14x14 --> 3x28x28
        self.deconv3 = nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2)
        
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
        return summary(self, (3, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        set_model_name(self, name, ckp_dir)

# =====================
# === Small Network ===
# =====================
class NetSmall(nn.Module):
    
    def __init__(self, name='NetSmall', ckp_dir='saved_models/cifar10/'):
        super(NetSmall, self).__init__()
        self.set_model_name(name, ckp_dir)
        
        # === Encoder ===
        # convolve 3x28x28 --> 6x28x28        
        self.conv1 = nn.Conv2d(3, 6, kernel_size=3, padding=1)
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
        # deconvolve 6x14x14 --> 3x28x28
        self.deconv3 = nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2)
        
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
        return summary(self, (3, 28, 28))
    
    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        load_checkpoint(self)        
        
    def set_model_name(self, name, ckp_dir):
        set_model_name(self, name, ckp_dir)


# ======================
# === Testing Models ===
# ======================
class Sreeni(nn.Module):
    
    def __init__(self, name='Sreeni', checkpoint_dir='saved_models/'):
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
        self.conv2 = nn.Conv2d(6, 4, kernel_size=3, padding=1)
        # downsample 8x16x16 --> 4x8x8
        self.down_pool2 = nn.MaxPool2d(2, 2)
        # flatten 8x8x8 --> 1x256
        # dense 1x256 --> 1x128
        self.fc1 = nn.Linear(256, 128)
        # dense 1x128 --> 1x24
        self.fc2 = nn.Linear(128, 24)
        
        # === Decoder ===
        # dense 1x24 --> 1x128
        self.fc3 = nn.Linear(24, 128)
        # dense 1x128 --> 1x256
        self.fc4 = nn.Linear(128, 256)        
        # reshape 1x256 --> 4x8x8
        # convolveT 4x8x8 --> 6x16x16
        self.conv3 = nn.ConvTranspose2d(4, 6, kernel_size=2, stride=2)
        # convolveT 6x16x16 --> 3x32x32
        self.conv4 = nn.ConvTranspose2d(6, 3, kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.decode(self.encode(x))
        return x
    
    def encode(self, x):
        # Encode x. Returns unactivated code. 
        x = self.down_pool1(F.relu(self.conv1(x)))
        x = self.down_pool2(F.relu(self.conv2(x)))
        # reshape
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def decode(self, x):
        # Decode x.
        x = F.relu(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        # reshape
        x = x.view(-1, 4, 8, 8)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x
        
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

# =========================
# === Utility Functions ===
# =========================

def set_model_name(model, name, ckp_dir):
    # Set model's name, and loaction of where to save state dict.
    model.name = name
    model.ckp_dir = ckp_dir
    model.ckp_file = os.path.join(model.ckp_dir, model.name+'_cifar10.pt')
        
    # Set device for model.
    model.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(model.device)
    
def save_checkpoint(model):
    print(f'=== Saving model {model.name} checkpoint === ')
    torch.save(model.state_dict(), model.checkpoint_file)

def load_checkpoint(model):
    print(f'=== Loading model {model.name} checkpoint ===')
    return torch.load(model.checkpoint_file)

