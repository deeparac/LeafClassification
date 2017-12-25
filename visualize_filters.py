import os
import sys

import mxnet as mx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img

from math import sqrt

from collections import namedtuple
Batch = namedtuple('Batch', ['data'])

# Function by gcalmettes from http://stackoverflow.com/questions/11159436/multiple-figures-in-a-single-window
def plot_figures(figures, nrows = 1, ncols=1, titles=False):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind,title in enumerate(sorted(figures.keys(), key=lambda s: int(s[3:]))):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        if titles:
            axeslist.ravel()[ind].set_title(title)

    for ind in range(nrows*ncols):
        axeslist.ravel()[ind].set_axis_off()

    if titles:
        plt.tight_layout()
    plt.show()


def get_dim(num):
    """
    Simple function to get the dimensions of a square-ish shape for plotting
    num images
    """

    s = sqrt(num)
    if round(s) < s:
        return (int(s), int(s)+1)
    else:
        return (int(s)+1, int(s)+1)

def visualize_filters(leafnet):
    # Load the best model
    sym, arg_params, aux_params = mx.model.load_checkpoint("../leafnet/chkpt", 200)

    nmr_array, img_array, img_label = leafnet.load_train_data()

    # Get the activation layers
    all_layers = sym.get_internals()
    viz_layer1 = all_layers['relu1_output']
    viz_layer2 = all_layers['relu2_output']
    viz_layers = [viz_layer1, viz_layer2]

    concated = all_layers['concat_output']

    # Pick random images to visualize
    img_to_visualize = np.random.choice(np.arange(0, len(img_array)), 1)
    img_tensor = mx.nd.array(img_array[img_to_visualize])
    img_to_show = img_array[img_to_visualize].reshape((96, 96))

    plt.title("Image used: #%d (digit=%d)" % 
              (img_to_visualize, img_label[img_to_visualize])
    )
    plt.imshow(img_to_show, cmap='gray')
    plt.tight_layout()
    plt.savefig('randomleaf.png')
    plt.show()
    for l, layer_sym in enumerate(viz_layers):
        viz_mod = mx.mod.Module(symbol=layer_sym, 
                                 context=mx.gpu(), 
                                 label_names=None, 
                                 data_names=['img_input']
        )
        viz_mod.bind(for_training=False, 
                      data_shapes=[('img_input', (1, 1, 96, 96))]
        )
        viz_mod.set_params(arg_params=arg_params, aux_params=aux_params)


        viz_mod.forward(Batch([img_tensor]))
        features = viz_mod.get_outputs()[0].asnumpy()

        for i, img in enumerate(features):
            print("Visualizing Convolutions Layer %d" % \
                  (l + 1)
            )
            fig_dict = {
                'flt{0}'.format(i): img[i, :, :] \
                    for i in range(len(features[0]))
            }
            plot_figures(fig_dict, *get_dim(len(fig_dict)))
            plt.savefig('../results/filter for layer' + ' ' + str(l + 1) + '.png')