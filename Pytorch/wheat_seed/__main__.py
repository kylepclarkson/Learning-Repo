import numpy as np
import torch

from matplotlib import pyplot as plt

def get_data_from_file(filepath, proportion=0.8):
    """
    :param filepath: File to load data from.
    :param proportion: The proportion of data that makes the training set.
        Remainder is proportion of validation set.
    :return: x_train, y_train, x_val, y_val
        y sets are encoded as onehot vectors.
    """
    data_np = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
    data = torch.from_numpy(data_np)
    # 210 rows 70,70,70 split for each class
    # split into training and validation sets
    n_samples = data.shape[0]
    n_train = int(proportion * n_samples)
    shuffle_indices = torch.randperm(n_samples)
    train_data = data[shuffle_indices[:n_train]].float()
    val_data = data[shuffle_indices[:-n_train]].float()

    # encode labels via one-hot vectors
    train_label = train_data[:, -1].long()
    val_label = val_data[:, -1].long()

    # there are 3 classes for this dataset.
    train_onehot = torch.zeros(train_data.shape[0], 3).float()
    val_onehot = torch.zeros(val_data.shape[0], 3).float()
    # subtract 1 as scatter_ assumes 0 based indexing;
    # data set uses labels: 1,2,3.
    train_onehot.scatter_(1, train_label.unsqueeze(1)-1, 1.0)
    val_onehot.scatter_(1, val_label.unsqueeze(1)-1, 1.0)
    return train_data[:, :-1], train_onehot, val_data[:, :-1], val_onehot

def plot_loss(lost_train_lst, lost_val_lst):
    plt.xlabel("epoch")
    plt.ylabel("loss")
    epoch_lst = np.arange(1, len(lost_val_lst)+1)
    plt.plot(epoch_lst, lost_train_lst)
    plt.plot(epoch_lst, lost_val_lst)

    plt.show()

def train_loop(x_train, y_train, x_val, y_val,
               model,
               criterion,
               optimizer,
               n_epochs,
               device):

    loss_train_lst = []
    loss_val_lst = []

    for epoch in range(1, n_epochs + 1):
        # Feed forward training data, compute loss.
        x_train.to(device=device)
        y_train.to(device=device)
        y_train_pred = model(x_train)
        loss_train = criterion(y_train_pred, y_train)

        # Feed forward validation data, compute loss.
        x_val.to(device=device)
        y_val.to(device=device)
        y_val_pred = model(x_val)
        loss_val = criterion(y_val_pred, y_val)

        # Added loss to lists for plotting.
        loss_train_lst.append(loss_train)
        loss_val_lst.append(loss_val)

        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f'Epoch: {epoch}. Training loss: {loss_train.item():.4f}. Validation loss: {loss_val.item():.4f}')

    return loss_train_lst, loss_val_lst

class TwoLayerNetwork(torch.nn.Module):
    def __init__(self, in_dim, h1_dim, out_dim):
        super(TwoLayerNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, h1_dim)
        self.linear2 = torch.nn.Linear(h1_dim, out_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        return x


class ThreeLayerNetwork(torch.nn.Module):

    def __init__(self, in_dim, h1_dim, h_2dim, out_dim):
        super(ThreeLayerNetwork, self).__init__()
        self.linear1 = torch.nn.Linear(in_dim, h1_dim)
        self.linear2 = torch.nn.Linear(h1_dim, h_2dim)
        self.linear3 = torch.nn.Linear(h_2dim, out_dim)

    def forward(self, x):
        x = torch.nn.functional.relu(self.linear1(x))
        x = torch.nn.functional.relu(self.linear2(x))
        x = torch.nn.functional.relu(self.linear3(x))
        return x

if __name__ == '__main__':
    x_train, y_train, x_val, y_val = get_data_from_file('wheat.csv', .85)

    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    print("Device: ", device)
    n_input = x_train.shape[1]
    n_out = y_train.shape[1]

    model = ThreeLayerNetwork(n_input, 128, 128, n_out)

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 15_000

    loss_train_lst, loss_val_lst = train_loop(x_train,
                                              y_train,
                                              x_val,
                                              y_val,
                                              model,
                                              criterion,
                                              optimizer,
                                              n_epochs,
                                              device)


    plot_loss(loss_train_lst, loss_val_lst)

