import matplotlib.pyplot as plt
import numpy as np
import multiprocessing


def imshow(img, text=None):
    """
    Plot image with text
    :param img: the image to plot
    :param text: add text to the image
    """
    img = img.numpy()
    plt.axis('off')
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


def get_workers():
    """
    Get suitable number of CPUs for training the model
    :return: number of CPUs
    """
    num_workers = multiprocessing.cpu_count()
    if num_workers > 1:
        num_workers -= 1
    return num_workers
