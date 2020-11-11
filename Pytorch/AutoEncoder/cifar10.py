# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 23:04:42 2020

@author: Kyle
"""
# %% Imports 
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import models.cifar_models
import utils

def train(n_epochs,
          model,
          optimizer,
          loss_function,
          train_loader,
          val_loader):
    print(f'Training model {model.name}. Device: {model.device}')
    best_val_loss = np.Infinity
    
    running_train_loss = []
    running_validation_loss = []

    for epoch in range(1, n_epochs+1):

        training_loss = 0.0

        # iterate over training batch
        for imgs, _ in train_loader:
            
            # move tensors to device
            imgs = imgs.to(model.device)

            # zero gradient.
            optimizer.zero_grad()
            # forward, backward, update
            imgs_out = model(imgs)
            loss = loss_function(imgs_out, imgs)
            
            loss.backward()
            optimizer.step()
            
            training_loss += loss.item()

        running_train_loss.append(training_loss)

        validation_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                # move tensors to device
                imgs = imgs.to(model.device)
                imgs_out = model(imgs)
    
                loss = loss_function(imgs_out, imgs)
                    
                validation_loss += loss.item()
        running_validation_loss.append(validation_loss)
            
        # print training and validation losses. 
        if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
            print(f'{datetime.datetime.now()} Epoch: {epoch}, Training loss: {training_loss:.3f}')
            print(f'{datetime.datetime.now()} Epoch: {epoch}, Validation loss: {validation_loss:.3f}')

            if validation_loss < best_val_loss:
                model.save_checkpoint()
                best_val_loss = validation_loss

    return running_train_loss, running_validation_loss

def test_model(model, val_set):
    # Sample random image from validation set. 
    # Display original and reconstruction. 
    idx = np.random.choice(len(val_set))
    x, _ = val_set[idx]
    y = model_out(model, x)
    imgs(x, y)

def model_out(model, img):
    model.eval()
    img = img.to(model.device)
    with torch.no_grad():
        y = model(img.unsqueeze(0)).squeeze(0)
        return y

def to_numpy(img, mean=0, std=1):
    img = img.cpu()
    npimg = img.numpy()
    return npimg*std + mean

def imgs(img1, img2, title1='Input', title2='Output', ):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text(title1)
    ax1.imshow(np.transpose(to_numpy(img1), (1,2,0)))

    ax2.title.set_text(title2)
    ax2.imshow(np.transpose(to_numpy(img2), (1,2,0)))

    plt.show()

def imshow(img, mean=0, std=1):
    npimg = img.numpy()
    npimg = npimg*std + mean
    # Display image by reordering channels to match pyplot's expectation
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def plot_loss(train_losses, val_losses, title=None, save_loc=None):
    plt.plot(np.arange(1, len(train_losses)+1), train_losses, color='orange', label='Training loss')
    plt.plot(np.arange(1, len(val_losses)+1), val_losses, color='blue', label='Validation loss')    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    if title is not None:
        plt.title(title)
    if save_loc is not None:
        plt.savefig(save_loc)
    plt.show()
    
# Get MNIST datasets
data_path = 'C:\\Users\\Kyle\\Documents\\GitHub\\data\\'

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

batch_size =  512
train_set = torchvision.datasets.CIFAR10(data_path, train=True, download=True,  transform=transform)
val_set = torchvision.datasets.CIFAR10(data_path, train=False, download=True,  transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


# %% Get model
# Set model, optimizer, loss function. 
loss_fn_name = 'L1Loss'
loss_fn = torch.nn.L1Loss()

# loss_fn_name = 'L2Loss'
# loss_fn = torch.nn.MSELoss()
# Set model, optimizer, loss function. 
# model = models.cifar_models.NetLargeDropout(name=f'NetLargeDropout-{loss_fn_name}')
# model = models.cifar_models.NetSmall(name=f'NetSmall-{loss_fn_name}')
# model = models.cifar_models.NetLarge(name=f'NetLarge-{loss_fn_name}')
# model = models.cifar_models.NetGrossDropout(name=f'NetGross-{loss_fn_name}')
# model = models.cifar_models.Net(name=f'Net-{loss_fn_name}')
model = models.cifar_models.NetTwo(name=f'Net2-{loss_fn_name}')
model.to(model.device)

# %% Train model
opt = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 1000
train_loss = []
val_loss = []
# Train model
train_loss_1, val_loss_1 = train(n_epochs, model, opt, loss_fn, train_loader, val_loader)
train_loss.extend(train_loss_1)
val_loss.extend(val_loss_1)
# Save train, val loss. 
title_name = f'{model.name}-{loss_fn_name}-Adam'
plot_loss(train_loss, val_loss, title=title_name, save_loc='plots/cifar10/'+title_name)
