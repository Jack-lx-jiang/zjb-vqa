import numpy as np
from keras import backend as K
from keras import initializers
from keras.engine.topology import Layer


# It a keras implementation of Action Vlad
# Too memory consuming and harder to converge than shallowFeatureModel....
# So it has not been tested on shallow feature

class VladPooling(Layer):

    def __init__(self, kmenas_init, alpha=1000, trainable=True, regularizer=None, **kwargs):
        self.kmenas_init = kmenas_init
        self.regularizer = regularizer
        self.alpha = alpha
        self.trainable = trainable
        self.supports_masking = True
        super(VladPooling, self).__init__(**kwargs)

    def build(self, input_shape):
        try:
            centers = int(self.kmenas_init)
            centers = np.random.normal(size=(centers, input_shape[-1]))
            print('Randomly initializing the {} netvlad cluster centers'.format(centers.shape))
        except ValueError:
            # with open(self.kmenas_init, 'rb') as fin:
            #     kmeans = pickle.load(fin)
            #     centers = kmeans.cluster_centers_
            centers = np.load(self.kmenas_init)
        self.num_centers = centers.shape[0]

        self.centers = self.add_weight(name='centers',
                                       shape=centers.shape,
                                       regularizer=self.regularizer,
                                       initializer=initializers.get('glorot_uniform'),
                                       trainable=self.trainable)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, 1) + centers.transpose().shape,
                                      regularizer=self.regularizer,
                                      initializer=initializers.get('glorot_uniform'),
                                      trainable=self.trainable)
        self.bias = self.add_weight(name='bias',
                                    shape=(centers.shape[0],),
                                    regularizer=self.regularizer,
                                    initializer=initializers.get('glorot_uniform'),
                                    trainable=self.trainable)
        self.set_weights(
            [centers, centers.transpose()[np.newaxis, np.newaxis, np.newaxis, ...] * 2 * self.alpha, -self.alpha *
             np.sum(np.square(centers), axis=1)])

        super(VladPooling, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, mask=None):
        conv_ouput = K.conv3d(x, self.kernel, strides=[1, 1, 1], padding='valid')
        dists = K.bias_add(conv_ouput, self.bias)
        assgn = K.softmax(dists, axis=-1)
        clusters = []
        for k in range(self.num_centers):
            cur_residuals = x - self.centers[k, :]
            cur_asgn_residuals = K.expand_dims(assgn[:, :, :, :, k]) * cur_residuals
            # cur_sum_residuals = K.sum(cur_asgn_residuals, axis=[1, 2, 3])
            cur_sum_residuals = K.mean(cur_asgn_residuals, axis=[1, 2, 3])
            clusters.append(cur_sum_residuals)
        k_clusters = K.stack(clusters, axis=1)

        return K.reshape(k_clusters, (K.shape(k_clusters)[0], -1))
        # intra_normed = K.l2_normalize(k_clusters, axis=2)
        # return K.reshape(intra_normed, (K.shape(intra_normed)[0], -1))
        # final_normed = K.l2_normalize(K.reshape(intra_normed, (K.shape(intra_normed)[0], -1)), axis=1)
        # return final_normed

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_centers * input_shape[-1])
