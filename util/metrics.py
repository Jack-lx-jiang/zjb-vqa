from keras import backend as K


def multians_accuracy(y_true, y_pred):
    pred = K.argmax(y_pred, axis=-1)
    return K.mean(K.in_top_k(y_true, pred, 3), axis=-1)
