# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:04:46 2020

@author: Kyle
"""
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import models.mnist_models
import utils

def train(n_epochs,
          model,
          optimizer,
          loss_function,
          train_loader,
          val_loader):
    print('Training. Device: ', model.device)
    best_val_loss = 0
    running_loss = []

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
                model.save_checkpoint()
                best_val_loss = validation_loss

    return running_loss

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

def plot_loss(training_epochs, losses):
    plt.plot(training_epochs, losses)
    plt.title('Training loss vs Epoch')
    

# Get MNIST datasets
data_path = 'C:\\Users\\Kyle\\Documents\\GitHub\\data\\'

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

batch_size = 8
train_set = torchvision.datasets.MNIST(data_path, train=True, download=True,  transform=transform)
val_set = torchvision.datasets.MNIST(data_path, train=False, download=True,  transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

# Set model, optimizer, loss function. 
# model = models.mnist_models.Chan()
model = models.mnist_models.ChanSmall()

model.to(model.device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
n_epochs = 50


loss = train(n_epochs, model, opt, loss_fn, train_loader, val_loader)

