import torch
import torch.nn as nn
import torch.nn.functional as F

class AppaNet(nn.Module):

    """
    TODO:
        - Define network (init)
            - Initialize parameters
        - Define forward function
        - Define loading/saving functions
    """
    def __init__(self, name, checkpoint_dir):
        super(AppaNet, self).__init__()

        self.name = name
        self.checkpoint_dir = checkpoint_dir

        # === Construct network ===
        # convolve: 1x28x28 --> 8x28x28
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # pool: 8x28x28 --> 8x14x14
        self.pool1 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout1 = nn.Dropout2d(p=0.3)

        # convolve: 8x14x14 --> 16x14x14
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # pool: 16x14x14 -->  16x7x7
        self.pool2 = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout2 = nn.Dropout(p=0.3)

        # fully connected: 16x7x7 --> 32x1
        self.dense1 = nn.Linear(16*7*7, 32)
        # fully connected: 32x1 --> 10x1
        self.dense2 = nn.Linear(32, 10)

        # Set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def forward(self, x):
        # convolve, activate, pool, dropout.
        x = self.dropout1(self.pool1(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool2(F.relu(self.conv2(x))))

        x = x.view(-1, 16*7*7)
        x = F.relu(self.dense1(x))
        x = self.dense2(x)

        return x

    def save_checkpoint(self):
        save_checkpoint(self)

    def load_checkpoint(self):
        self.load_state_dict(load_checkpoint(self))


def save_checkpoint(model):
    print(f'=== Saving model {model.name} checkpoint === ')
    torch.save(model.state_dict(), model.checkpoint_dir)

def load_checkpoint(model):
    print(f'=== Loading model {model.name} checkpoint ===')
    return torch.load(model.checkpoint_file)

