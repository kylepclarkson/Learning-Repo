# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 12:04:46 2020

@author: Kyle
"""
# %% Imports
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
          reconstruction_loss_fn,
          cluster_loss_fn,
          train_loader,
          val_loader):
    print(f'Training model {model.name}. Device: {model.device}')
    best_val_loss = np.Infinity
    
    running_train_loss = []
    running_validation_loss = []

    for epoch in range(1, n_epochs+1):

        training_loss = 0.0

        # iterate over training batch
        for imgs, labels in train_loader:
            
            # move tensors to device
            imgs = imgs.to(model.device)

            # zero gradient.
            optimizer.zero_grad()
            # forward, backward, update
            imgs_out = model(imgs)
            loss = reconstruction_loss_fn(imgs_out, imgs)
            if cluster_loss_fn is not None:
                imgs_embedding = model.encode(imgs)
                loss += cluster_loss_fn(imgs_embedding, labels)
            
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
    
                loss = reconstruction_loss_fn(imgs_out, imgs)
                    
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

def random_subset(in_set, n):
    # Sample n unique and random samples from in_set
    # Return list of 2-tuples (data, label).
    indices = np.random.choice(np.arange(len(in_set)), size=n, replace=False)
    imgs = [in_set[i][0] for i in indices]
    labels = [in_set[i][1] for i in indices]
    return imgs, labels

def test_model(model, val_set):
    # Sample random image from validation set. 
    # Display original and reconstruction. 
    idx = np.random.choice(len(val_set))
    x, _ = val_set[idx]
    y = model_out(model, x)
    imgs(x, y)

def model_code(model, img):
    # Get code for img.
    model.eval()
    img = img.to(model.device)
    with torch.no_grad():
        code = model.encode(img.unsqueeze(0))
        return code
def model_out(model, img):
    # Get model output (reconstruction) for img.
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

def cluster_loss_fn(x, y):
    """
    Parameters
    ----------
    x : PyTorch Tensor of size BxD where B is the batchsize and D the dimension
        of the encoding.
    y : PyTorch Tensor of size B where B is the batchsize. 

    Returns
    -------
    None.

    """
    # TODO implement.
    pass

# Get MNIST datasets
data_path = 'C:\\Users\\Kyle\\Documents\\GitHub\\data\\'

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

batch_size =  512
train_set = torchvision.datasets.MNIST(data_path, train=True, download=True,  transform=transform)
val_set = torchvision.datasets.MNIST(data_path, train=False, download=True,  transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)


# %% Get model
# Set model, optimizer, loss function. 
loss_fn_name = 'L1Loss'
reconstruction_loss_fn = torch.nn.L1Loss()
cluster_loss_fn = None

# loss_fn_name = 'L2Loss'
# loss_fn = torch.nn.MSELoss()

# model = models.mnist_models.NetLargeDropout(name=f'NetLargeDropout-{loss_fn_name}')
model = models.mnist_models.NetSmall(name=f'NetSmall-{loss_fn_name}')
# model = models.mnist_models.NetLarge(name=f'NetLarge-{loss_fn_name}')

model.to(model.device)

# %% Train model
opt = torch.optim.Adam(model.parameters(), lr=0.001)
n_epochs = 1000
train_loss = []
val_loss = []
# Train model
train_loss_1, val_loss_1 = train(n_epochs,
                                 model,
                                 opt,
                                 reconstruction_loss_fn=reconstruction_loss_fn,
                                 cluster_loss_fn=cluster_loss_fn,
                                 train_loader=train_loader,
                                 val_loader=val_loader)
train_loss.extend(train_loss_1)
val_loss.extend(val_loss_1)
# Save train, val loss. 
title_name = f'{model.name}-{loss_fn_name}-Adam'
plot_loss(train_loss, val_loss, title=title_name, save_loc='plots/cifar10/'+title_name)

# %% View embedding
n_points = 400
# Get data, labels from dataset.
imgs, labels = random_subset(val_set, n_points)

# Encode imgs to code using model, stack into 2D array.
encoded_imgs = np.vstack([model_code(model, img).cpu().numpy() for img in imgs])

utils.display_tsne(encoded_imgs, labels)