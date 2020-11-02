import os
import datetime
import numpy as np
import torch
import torchvision

import models.models
import train_utils as utils

def training_loop(n_epochs,
                  model,
                  optimizer,
                  loss_function,
                  train_loader,
                  val_loader):
    print('Training:')
    best_val_loss = 0
    running_loss = []

    for epoch in range(1, n_epochs+1):

        training_loss = 0.0

        # iterate over training batch
        for imgs, _ in train_loader:
            # move tensors to device
            imgs = imgs.to(model.device)
            imgs_out = model(imgs)

            loss = loss_function(imgs_out, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        running_loss.append(training_loss)

        # print training and validation losses. save if model achieve better accuracy rate.
        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            validation_loss = 0.0
            for imgs, _ in val_loader:
                # move tensors to device
                imgs = imgs.to(model.device)

                imgs_out = model(imgs)

                loss = loss_function(imgs_out, imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                validation_loss += loss.item()

            print(f'{datetime.datetime.now()} Epoch: {epoch}, Training loss: {training_loss:.3f}')
            print(f'{datetime.datetime.now()} Epoch: {epoch}, Validation loss: {validation_loss:.3f}')

            if validation_loss > best_val_loss:
                # save model
                # print('Saving model!')
                model.save_checkpoint()
                best_val_loss = validation_loss

    return running_loss

def get_datasets(data_path, batch_size=32):
    # Get CIFAR10 training and validation datasets and loaders.

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset_cifar = torchvision.datasets.CIFAR10(data_path, transform=transform, train=True, download=True)
    validset_cifar = torchvision.datasets.CIFAR10(data_path, transform=transform, train=False, download=True)
    training_loader = torch.utils.data.DataLoader(trainset_cifar, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validset_cifar, batch_size=batch_size, shuffle=False)

    return trainset_cifar, validset_cifar, training_loader, validation_loader

def train_model(n_epochs,
                model,
                optimizer,
                loss_function,
                training_loader,
                validation_loader):


    running_loss = training_loop(n_epochs=n_epochs,
                                model=model,
                                optimizer=optimizer,
                                loss_function=loss_function,
                                train_loader=training_loader,
                                val_loader=validation_loader)
    return running_loss

if __name__ == '__main__':

    # Get CIFAR10 dataset.
    data_path = 'C:\\Users\\Kyle\\Documents\\GitHub\\data\\'
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Create model
    model = models.models.LittmanNet(name='LittmanNet')
    model.to(model.device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_function = torch.nn.L1Loss()
    n_epochs = 20

    train_set, val_set, train_loader, val_loader = get_datasets(data_path, batch_size=16)
    # train_model(n_epochs=n_epochs,
    #             model=model,
    #             optimizer=optimizer,
    #             loss_function=loss_function,
    #             training_loader=train_loader,
    #             validation_loader=val_loader)

    # model.load_checkpoint()
    # model.eval()

    # _, valset, _, _ = get_datasets(data_path)

    # with torch.no_grad():
    #     img, _ = valset[0]
    #     img = img.to(model.device)
    #     # utils.imshow(img.cpu())

    #     img_out = model(img.unsqueeze(0)).squeeze(0)
    #     print(f'Img1 shape, img2 shape: {img.shape}, {img_out.shape}')
    #     utils.plot_side_by_side(img.cpu(), 'Original Image', img_out.cpu().clamp(0, 1), 'Generated Image')


