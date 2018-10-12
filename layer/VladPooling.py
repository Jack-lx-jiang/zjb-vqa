import pickle

import numpy as np
from keras import backend as K
from keras.engine.topology import Layer


class VladPooling(Layer):

    def __init__(self, kmenas_init, regularizer=None, **kwargs):
        self.kmenas_init = kmenas_init
        self.regularizer = regularizer
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        try:
            centers = int(self.kmenas_init)
            centers = np.random.normal(size=(centers, input_shape[-1]))
            print('Randomly initializing the {} netvlad cluster centers'.format(centers.shape))
        except ValueError:
            with open(self.kmenas_init, 'rb') as fin:
                kmeans = pickle.load(fin)
                centers = kmeans.cluster_centers_

        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1) + centers.transpose().shape,
                                      regularizer=self.regularizer)
        self.bias = self.add_weight(name='bias',
                                    shape=centers.shape[0])

        super(VladPooling, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return K.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
