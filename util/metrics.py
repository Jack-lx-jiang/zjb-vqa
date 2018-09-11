from keras import backend as K


def multians_accuracy(y_true, y_pred):
    pred = K.argmax(y_pred, axis=-1)
    mask = K.one_hot(pred, K.shape(y_true)[1])
    res = K.sum(mask * y_true, axis=-1)
    return K.mean(res)
