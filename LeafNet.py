import os
import sys
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder, StandardScaler

import logging
logging.getLogger().setLevel(logging.DEBUG)

class LeafNet(object):
    def __init__(self, root_path='../input/images/', height=96, width=96, seed=2017,
                 ctx=mx.gpu(), batch_size=32, epochs=200, learning_rate=.001, drop_prob=.5,
                 momentum=.9, weight_decay=.001):
        
        self.root_path = root_path
        self.height, self.width = height, width
        self.size = (self.height, self.width)
        self.max_dim = max(height, width)
        
        mx.random.seed(seed)
        self.ctx = ctx
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.drop_prob = drop_prob
        self.out, self.mod = self.build_model()
        self.trainiter, self.valiter, self.testiter \
                    = self.train_val_test_split()
        self.chkpt_prefix = '../leafnet/chkpt'
        self.load_epoch = epochs
    
    def load_numeric_training(self):
        train = pd.read_csv('../input/train.csv')
        ID = train.pop('id')
        y = train.pop('species')
        y = LabelEncoder().fit(y).transform(y)
        X = StandardScaler().fit(train).transform(train)

        ID_list = ID.values.tolist()

        species = dict()
        for idx, val in enumerate(ID_list):
            species[val] = y[idx]

        return ID, X, y, species

    def load_numeric_test(self):
        test = pd.read_csv('../input/test.csv')
        ID = test.pop('id')
        test = StandardScaler().fit(test).transform(test)
        return ID, test
    
    def resize_image(self, img):
        """
        Resize the image to so the maximum side is of size max_dim
        Returns a new image of the right size
        """
        # Get the axis with the larger dimension
        max_ax = max((0, 1), key=lambda i: img.size[i])
        # Scale both axes so the image's largest dimension is max_dim
        scale = self.max_dim / float(img.size[max_ax])
        return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


    def readin_image(self, ids, center=True):
        """
        Takes as input an array of image ids and loads the images as numpy
        arrays with the images resized so the longest side is max-dim length.
        If center is True, then will place the image in the center of
        the output array, otherwise it will be placed at the top-left corner.
        """
        # Initialize the output array
        X = np.empty((len(ids), self.max_dim, self.max_dim, 1))
        for i, idee in enumerate(ids):
            # Turn the image into an array
            x = self.resize_image(load_img(os.path.join('../input', 'images', str(idee) + '.jpg'), grayscale=True))
            x = img_to_array(x)
            # Get the corners of the bounding box for the image
            length = x.shape[0]
            width = x.shape[1]
            if center:
                h1 = int((self.max_dim - length) / 2)
                h2 = h1 + length
                w1 = int((self.max_dim - width) / 2)
                w2 = w1 + width
            else:
                h1, w1 = 0, 0
                h2, w2 = (length, width)
            # Insert into image matrix
            # NOTE: Theano users comment line below and
            X[i, h1:h2, w1:w2, 0:1] = x
            # X[i, 0:1, h1:h2, w1:w2] = x  # uncomment this
        # Scale the array values so they are between 0 and 1
        scaled =  np.around(X / 255.0)
        img_array = []
        for im in scaled:
            im = np.squeeze(im)
            im = np.expand_dims(im, axis=0)
            img_array.append(im)
        
        out = np.array(img_array)
        return out
            
    def load_train_data(self):
        ID, X, y, species = self.load_numeric_training()
        ids = ID.values.tolist()
        img_array = self.readin_image(ids)
        img_label = np.array([species[id] for id in ids])
        nmr_array = np.array(X.copy().tolist())
        return nmr_array, img_array, img_label
    
    def build_model(self):
        img_input = mx.sym.Variable('img_input')
        numerical = mx.sym.Variable('numerical')
        conv1 = mx.sym.Convolution(data = img_input,
                                   kernel = (5, 5),
                                   stride = (1, 1),
                                   num_filter = 8,
                                   name = 'conv1')
        bn1 = mx.sym.BatchNorm(data = conv1, 
                               name = 'BatchNorm1')
        relu1 = mx.sym.Activation(data = bn1,
                                  act_type = 'relu',
                                  name = 'relu1')
        pool1 = mx.sym.Pooling(data = relu1,
                               pool_type = 'max',
                               kernel = (2, 2),
                               stride = (2, 2),
                               name = 'pool1')

        conv2 = mx.sym.Convolution(data = pool1,
                                   kernel = (5, 5),
                                   stride = (1, 1),
                                   num_filter = 32,
                                   name = 'conv2')
        bn2 = mx.sym.BatchNorm(data = conv2, 
                               name = 'BatchNorm2')
        relu2 = mx.sym.Activation(data = bn2,
                                  act_type = 'relu',
                                  name = 'relu2')
        pool2 = mx.sym.Pooling(data = relu2,
                               pool_type = 'max',
                               kernel = (2, 2),
                               stride = (2, 2),
                               name = 'pool2')
        flatten = mx.sym.Flatten(data = pool2,
                                 name = 'flatten')

        concat = mx.sym.concat(flatten, numerical, 
                               dim=1, 
                               name = 'concat')
        fc1 = mx.sym.FullyConnected(data = concat, 
                                    num_hidden = 1500,
                                    name = 'fc1')
        dpout = mx.sym.Dropout(data = fc1,
                             p = self.drop_prob,
                             mode = 'training',
                             name = 'dropout')
        fc2 = mx.sym.FullyConnected(data = dpout, 
                                    num_hidden = 99,
                                    name = 'fc2')
        out = mx.sym.SoftmaxOutput(data = fc2,
                                   name = 'softmax')

        mod = mx.mod.Module(symbol = out, 
                            context = self.ctx,
                            data_names = ['img_input', 'numerical'])
        return out, mod
    
    def visualize_cnn(self):
        mx.viz.plot_network(symbol = self.out)

    def train_val_test_split(self):
        nmr_array, img_array, img_label = self.load_train_data()
        img_train = img_array[:int(.7*len(nmr_array)), :]
        nmr_train = nmr_array[:int(.7*len(nmr_array)), :]
        lbl_train = img_label[:int(.7*len(nmr_array))]
        
        img_val = img_array[int(.7*len(nmr_array)):int(.9*len(nmr_array)), :]
        nmr_val = nmr_array[int(.7*len(nmr_array)):int(.9*len(nmr_array)), :]
        lbl_val = img_label[int(.7*len(nmr_array)):int(.9*len(nmr_array))]
        
        img_test = img_array[int(.9*len(nmr_array)):, :]
        nmr_test = nmr_array[int(.9*len(nmr_array)):, :]
        lbl_test = img_label[int(.9*len(nmr_array)):]

        traindata = {
            'img_input': img_train,
            'numerical': nmr_train
        }
        valdata = {
            'img_input': img_val,
            'numerical': nmr_val
        }
        testdata = {
            'img_input': img_test,
            'numerical': nmr_test
        }

        trainiter = mx.io.NDArrayIter(data = traindata, 
                                      label = lbl_train, 
                                      batch_size = self.batch_size)
        valiter = mx.io.NDArrayIter(data = valdata, 
                                    label = lbl_val, 
                                    batch_size = self.batch_size)
        testiter = mx.io.NDArrayIter(data = testdata, 
                                     label = lbl_test, 
                                     batch_size = self.batch_size)
        return trainiter, valiter, testiter

    def cnn_fit(self):
        self.mod.fit(train_data = self.trainiter, 
                     eval_data  = self.valiter,
                     optimizer = 'RMSProp',
                     optimizer_params = {
                         'learning_rate': self.learning_rate,
                         'wd': self.weight_decay
                     },
                     eval_metric = ['accuracy',
                                   'ce'],
                     num_epoch = self.epochs)

    def cnn_eval(self):
        score = self.mod.score(self.testiter, ['acc', 'ce'])
        print(score)
    
    def get_fulltrain_iter(self):
        nmr_array, img_array, label = self.load_train_data()
        data = {
            'img_input': img_array,
            'numerical': nmr_array
        }
        return mx.io.NDArrayIter(data = data,
                                 label = label,
                                 batch_size = self.batch_size)
    def fulltrain_fit(self):
        train_iter = self.get_fulltrain_iter()
        self.executor = self.mod.bind(
            data_shapes = train_iter.provide_data,
            label_shapes = train_iter.provide_label
        )
        self.mod.init_params()
        self.mod.fit(train_data = train_iter,
                     optimizer = 'RMSProp',
                     optimizer_params = {
                         'learning_rate': self.learning_rate,
                         'wd': self.weight_decay
                     },
                     eval_metric = ['accuracy',
                                   'ce'],
                     num_epoch = self.epochs,
                     epoch_end_callback = mx.callback.do_checkpoint(
                             prefix = self.chkpt_prefix,
                             period = 50
                     )
#                      batch_end_callback = mx.callback.Speedometer(
#                          batch_size = self.batch_size,
#                          frequent = 10
#                      )
        )
    
    def load_leafnet(self, epoch=None):
        epoch = self.load_epoch
        chkpt = "../leafnet/chkpt"
        mx.model.load_checkpoint(chkpt, epoch)
    
    def predict(self):
        ID, nmr = self.load_numeric_test()
        img = self.readin_image(ID.values.tolist())
        data = {
            'img_input': img,
            'numerical': nmr
        }
        test_iter = mx.io.NDArrayIter(data = data, 
                                      batch_size = self.batch_size
        )
        probs = self.mod.predict(test_iter)
        probs = probs.asnumpy()
        return probs
