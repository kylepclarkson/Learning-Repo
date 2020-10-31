import numpy as np
import matplotlib.pyplot as plt

def plot_side_by_side(img1, title1, img2, title2):

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.title.set_text('Input Image')
    ax1.imshow(np.transpose(img1,(1,2,0)))

    ax2.title.set_text('Output Image')
    ax2.imshow(np.transpose(img2,(1,2,0)))

    plt.show()

def imshow(img):
    npimg = img.numpy()
    # Display image by reordering channels to match pyplot's expectation
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

def plot_loss(training_epochs, losses):
    plt.plot(training_epochs, losses)
    plt.title('Training loss vs Epoch')
