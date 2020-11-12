import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def plot_side_by_side(img1, title1, img2, title2):

    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.title.set_text('Input Image')
    ax1.imshow(np.transpose(img1,(1,2,0)))

    ax2.title.set_text('Output Image')
    ax2.imshow(np.transpose(img2,(1,2,0)))

    plt.show()

def imshow(img, mean=1, std=1):
    npimg = img.numpy()
    npimg = npimg*std + mean
    # Display image by reordering channels to match pyplot's expectation
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def plot_loss(training_epochs, losses):
    plt.plot(training_epochs, losses)
    plt.title('Training loss vs Epoch')


def display_tsne(data_points, labels):
    # Embed list of points into 2D space using t-distributed Stochastic Neighbour embedding.
    # data_points must numpy array.
    embedded = TSNE(n_components=2).fit_transform(data_points)
    
    # plot data points.
    fig, ax = plt.subplots()
    scatter = ax.scatter(embedded[:,0], embedded[:, 1], c=labels)
    
    legend_classes = ax.legend(*scatter.legend_elements(),
                               title="Digits",
                               loc="center left",
                               bbox_to_anchor=(1, 0.5))
    # plt.add_artist(legend_classes)
    ax.add_artist(legend_classes)
    
    plt.show()