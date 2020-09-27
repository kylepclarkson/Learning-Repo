import numpy as np
import torch

def get_data_from_file(filepath):
    """
    :param filepath: File to load data from.
    :return: Two PyTorch tensors containing dataset features and labels.
        Labels are encoded as onehot vectors.
    """
    data_np = np.loadtxt(filepath, dtype=np.float32, delimiter=',')
    data = torch.from_numpy(data_np)
    print(data)
    labels = data[:, -1].long()
    # there are 3 classes for this dataset.
    onehot = torch.zeros(labels.shape[0], 3).long()
    # subtract 1 as scatter_ assumes 0 based indexing;
    # data set uses labels: 1,2,3.
    onehot.scatter_(1, labels.unsqueeze(1)-1, 1.0)
    return data[:, :-1], onehot


if __name__ == '__main__':
    x, y = get_data_from_file('wheat.csv')

    print(x.shape, y.shape)
    print(x)
    print(y)