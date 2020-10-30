import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision

import models

def training_loop(n_epochs,
                  model,
                  optimizer,
                  loss_function,
                  train_loader,
                  val_loader):
    print('Training:')
    best_val_accuracy = 0
    running_loss = []

    for epoch in range(1, n_epochs+1):

        training_loss = 0.0

        # iterate over training batch
        for imgs, labels_true in train_loader:
            # move tensors to device
            imgs = imgs.to(model.device)
            labels_true = labels_true.to(model.device)

            labels_pred = model(imgs)

            loss = loss_function(labels_pred, labels_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        running_loss.append(training_loss)
        # print training and validation losses. save if model achieve better accuracy rate.
        if epoch % 10 == 0 or epoch == 1 or epoch == n_epochs:
            loss = training_loss / len(train_loader)
            print(f'{datetime.datetime.now()} Epoch: {epoch}, Training loss: {loss:.3f}')

            training_accuracy = validate_model(model, train_loader)
            validation_accuracy = validate_model(model, val_loader)
            print(f'Training accuracy: {training_accuracy:.3f}')
            print(f'Validation accuracy: {validation_accuracy:.3f}')

            if validation_accuracy > best_val_accuracy:
                # save model
                print('Saving model!')
                model.save_checkpoint()
                best_val_accuracy = validation_accuracy

    return running_loss

def validate_model(model, data_loader):
    # Count number of correctly predicted data items, divided by number of items.
    correct_pred, total_data = 0, 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(model.device)
            labels = labels.to(model.device)

            labels_pred = model(imgs)
            _, predicted = torch.max(labels_pred, dim=1)
            total_data += labels.shape[0]
            correct_pred += int((predicted == labels).sum())
    return correct_pred / total_data

def plot_loss(training_epochs, losses, figure_file):
    plt.plot(training_epochs, losses)
    plt.title('Training loss vs Epoch')
    plt.savefig(figure_file)

if __name__ == '__main__':

    # TODO apply transform to normalize data
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    # Get mnist data
    data_path = 'mnist_data/'
    training_set = torchvision.datasets.MNIST(data_path, train=True, download=True, transform=transforms)
    training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

    validation_set = torchvision.datasets.MNIST(data_path, train=False, download=True, transform=transforms)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)

    model = models.AppaNet()
    print('Model parameters: ', sum(p.numel() for p in model.parameters()))

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = torch.nn.CrossEntropyLoss()
    n_epochs = 50

    running_loss = training_loop(n_epochs=n_epochs,
                                model=model,
                                optimizer=opt,
                                loss_function=loss_function,
                                train_loader=training_loader,
                                val_loader=validation_loader)

    plot_loss(np.arange(1, n_epochs+1), running_loss, model.name)

