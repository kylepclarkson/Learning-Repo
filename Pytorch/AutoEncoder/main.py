import torch

import models.models

if __name__ == '__main__':

    model = models.models.LittmanNet()
    model.cuda()
    x = torch.zeros(1, 3, 32, 32)
    x = x.to(model.device)
    print('model device: ', model.device)
    print('x type: ', x.type)
    print(f'x shape: {x.shape}')

    v = model.encode(x)
    y = model(x)

    print('v shape: ', v.shape)
    print('y shape: ', y.shape)

