import numpy as np
import tensorflow as tf
from keras import backend as K

'''
Compatible with tensorflow backend
'''


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=-1) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)

    return focal_loss_fixed


def focal_loss_sum(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed


def focal_loss_mean(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=-1) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)

    return focal_loss_fixed


def focal_loss_weighted(gamma=2., alpha=.25, reverse_ans=None, bias=0):
    # print(reverse_ans.shape)
    reverse_ans *= 25
    reverse_ans += bias
    print(reverse_ans)
    # print(reverse_ans.shape)
    ra = K.variable(reverse_ans)
    ra = K.expand_dims(ra, 0)

    def focal_loss_fixed(y_true, y_pred):
        # np.set_printoptions(threshold=np.inf)
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        # print(y_true.shape, 'yture')
        ans_reverse_frequence = y_true * ra
        # ans_reverse_frequence = K.print_tensor(ans_reverse_frequence, 'ans_reverse_frequence')
        # ans_reverse_frequence = y_true*reverse_ans[np.newaxis,:]
        # ans_reverse_frequence = tf.where(tf.equal(y_true, 1), ra, tf.zeros_like(y_pred))
        y_sum = tf.reduce_sum(y_true, axis=-1, keepdims=True)
        # y_sum = K.print_tensor(y_sum, 'y sum')
        ans_softmax = K.softmax(ans_reverse_frequence)
        # ans_softmax = K.print_tensor(ans_softmax, 'softmax')
        ans_softmax_max = K.print_tensor(K.max(ans_softmax, -1), 'max')
        ans_weight = ans_softmax * y_sum

        ans_weight_max = K.print_tensor(K.max(ans_weight, -1), 'max')
        # ans_weight = K.print_tensor(ans_weight, 'ans_weight')
        # print(ans_weight.shape)

        # sum = K.sum(ans_weight, axis=-1)
        # sum = tf.Print(sum, [sum], 'class weigth sum')
        # ans_weight = K.print_tensor(ans_weight, 'class weigth sum')

        pos_loss = -K.sum(alpha * ans_weight * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=-1)
        # pos_loss_mean = K.print_tensor(K.mean(pos_loss), 'pos_loss')
        neg_loss = - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)
        # neg_loss_mean = K.print_tensor(K.mean(neg_loss), 'neg_loss')
        # return pos_loss + neg_loss + 0* pos_loss_mean +0*neg_loss_mean
        # return pos_loss + neg_loss + ans_softmax_max*0
        return pos_loss + neg_loss

    return focal_loss_fixed


def focal_loss_with_entropy(gamma=2., alpha=.25, ans_entropy=None):
    ans_entropy = ans_entropy * 20
    print(ans_entropy)
    print(np.max(ans_entropy))
    # print(ans_entropy.shape)
    ae = K.variable(ans_entropy)
    ae = K.expand_dims(ae, 0)

    def focal_loss_fixed(y_true, y_pred):
        # np.set_printoptions(threshold=np.inf)
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        # # print(y_true.shape, 'yture')
        # ans_reverse_frequence = y_true*ra
        # # ans_reverse_frequence = K.print_tensor(ans_reverse_frequence, 'ans_reverse_frequence')
        # # ans_reverse_frequence = y_true*reverse_ans[np.newaxis,:]
        # # ans_reverse_frequence = tf.where(tf.equal(y_true, 1), ra, tf.zeros_like(y_pred))
        # y_sum = tf.reduce_sum(y_true, axis=-1,keepdims=True)
        # # y_sum = K.print_tensor(y_sum, 'y sum')
        # ans_softmax = K.softmax(ans_reverse_frequence)
        # # ans_softmax = K.print_tensor(ans_softmax, 'softmax')
        # ans_softmax_max = K.print_tensor(K.max(ans_softmax, -1), 'max')
        # ans_weight = ans_softmax*y_sum
        #
        # ans_weight_max = K.print_tensor(K.max(ans_weight , -1), 'max')
        # ans_weight = K.print_tensor(ans_weight, 'ans_weight')
        # print(ans_weight.shape)

        # sum = K.sum(ans_weight, axis=-1)
        # sum = tf.Print(sum, [sum], 'class weigth sum')
        # ans_weight = K.print_tensor(ans_weight, 'class weigth sum')

        pos_loss = -K.sum(alpha * ae * K.pow(1. - pt_1, gamma) * K.log(pt_1), axis=-1)
        pos_loss_mean = K.print_tensor(K.mean(pos_loss), 'pos_loss')
        neg_loss = - K.sum((1 - alpha) * ae * K.pow(pt_0, gamma) * K.log(1. - pt_0), axis=-1)
        neg_loss_mean = K.print_tensor(K.mean(neg_loss), 'neg_loss')
        # return pos_loss + neg_loss + 0* pos_loss_mean +0*neg_loss_mean
        # return pos_loss + neg_loss + ans_softmax_max*0
        return pos_loss + neg_loss

    return focal_loss_fixed


def focal_loss_sum_with_entropy(gamma=2., alpha=.25, ans_entropy=None):
    ans_entropy = ans_entropy * 20
    print(ans_entropy)
    print(np.max(ans_entropy))
    # print(ans_entropy.shape)
    ae = K.variable(ans_entropy)
    ae = K.expand_dims(ae, 0)

    def focal_loss_fixed(y_true, y_pred):
        # np.set_printoptions(threshold=np.inf)
        eps = 1e-12
        y_pred = K.clip(y_pred, eps, 1. - eps)

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        pos_loss = -K.sum(alpha * ae * K.pow(1. - pt_1, gamma) * K.log(pt_1))
        pos_loss_mean = K.print_tensor(K.mean(pos_loss), 'pos_loss')
        neg_loss = - K.sum((1 - alpha) * ae * K.pow(pt_0, gamma) * K.log(1. - pt_0))
        neg_loss_mean = K.print_tensor(K.mean(neg_loss), 'neg_loss')
        # return pos_loss + neg_loss + 0* pos_loss_mean +0*neg_loss_mean
        # return pos_loss + neg_loss + ans_softmax_max*0
        return pos_loss + neg_loss

    return focal_loss_fixed
