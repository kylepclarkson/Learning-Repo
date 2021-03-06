import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class AppaNet(nn.Module):

    def __init__(self, name='appa', checkpoint_dir='models/'):
        super(AppaNet, self).__init__()

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_mnist.pt')

        # === Construct network ===
        # convolve: 1x28x28 --> 4x28x28
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        # pool: 4x28x28 --> 8x14x14
        self.pool1 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout1 = nn.Dropout2d(p=0.3)

        # convolve: 4x14x14 --> 8x14x14
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)
        # pool: 8x14x14 -->  8x7x7
        self.pool2 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout2 = nn.Dropout(p=0.3)

        # fully connected: 8x7x7 --> 20x1
        self.dense1 = nn.Linear(8*7*7, 20)
        # fully connected: 32x1 --> 10x1
        self.dense2 = nn.Linear(20, 10)

        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        # convolve, activate, pool, dropout.
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))

        x = x.view(-1, 8*7*7)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        self.load_state_dict(load_checkpoint(self))


def save_checkpoint(model):
    print(f'=== Saving model {model.name} checkpoint === ')
    torch.save(model.state_dict(), model.checkpoint_file)

def load_checkpoint(model):
    print(f'=== Loading model {model.name} checkpoint ===')
    return torch.load(model.checkpoint_file)

