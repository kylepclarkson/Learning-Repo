import numpy as np
import torch

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
    train_data = data[shuffle_indices[:n_train]]
    val_data = data[shuffle_indices[:-n_train]]

    # encode labels via one-hot vectors
    train_label = train_data[:, -1].long()
    val_label = val_data[:, -1].long()

    # there are 3 classes for this dataset.
    train_onehot = torch.zeros(train_data.shape[0], 3).long()
    val_onehot = torch.zeros(val_data.shape[0], 3).long()
    # subtract 1 as scatter_ assumes 0 based indexing;
    # data set uses labels: 1,2,3.
    train_onehot.scatter_(1, train_label.unsqueeze(1)-1, 1.0)
    val_onehot.scatter_(1, val_label.unsqueeze(1)-1, 1.0)
    return train_data[:, :-1], train_onehot, val_data[:, :-1], val_onehot

class SimpleModel(torch.nn.Module):

    def __init__(self, in_dim, h1_dim, h_2dim, out_dim):
        super(SimpleModel, self).__init__()
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

    print(x_train.shape, y_train.shape)
    print(x_val.shape, y_val.shape)