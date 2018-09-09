import pickle as p

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Masking, GRU, RepeatVector, Concatenate, Softmax, multiply, Lambda
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adadelta

from dataset import Dataset

dataset = Dataset()
batch_size = 128


def base_model(video, question):
    embedding_size = 300
    embedding_layer = Embedding(dataset.vocabulary_size, embedding_size, input_length=dataset.max_question_len,
                                mask_zero=True)(question)
    question_encoding = GRU(300)(Masking()(embedding_layer))
    qe_repeat = RepeatVector(dataset.max_video_len)(question_encoding)
    video_question = Concatenate()([video, qe_repeat])
    attention_dense1 = Dense(512)(video_question)
    attention_dense2 = Dense(1)(attention_dense1)
    attention_score = Softmax(axis=-2)(attention_dense2)
    video_encoding = Lambda(lambda x: K.sum(x, axis=-2))(multiply([video, attention_score]))
    video_encoding2 = Dense(512)(video_encoding)
    question_encoding2 = Dense(512)(question_encoding)
    combine_encoding = multiply([video_encoding2, question_encoding2])
    decode = Dense(2048)(combine_encoding)
    logit = Dense(dataset.answer_size, activation='sigmoid')(decode)
    return logit


def multians_accuracy(y_true, y_pred):
    pred = K.argmax(y_pred, axis=-1)
    return K.mean(K.in_top_k(y_true, pred, 3),axis=-1)

video = Input((dataset.max_video_len, dataset.frame_size))
question = Input((dataset.max_question_len,), dtype='int32')
model = Model(inputs=[video, question], outputs=base_model(video, question))
model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
exp_name = 'experiments/baseline_test'


nb_step = 3325 * 5 // batch_size + 1
train_mod = False
if train_mod:
    trained = model.fit_generator(dataset.generator(batch_size, 'train'), nb_step, 100,
                                  validation_data=dataset.generator(batch_size, 'train'),
                                  validation_steps=nb_step, callbacks=[EarlyStopping(patience=5),
                                                                       ModelCheckpoint(
                                                                           exp_name + '_.{epoch:02d}-{val_loss:.2f}.pkl',
                                                                           save_best_only=True)])
    p.dump(trained.history, open(exp_name + '_history.pkl', 'wb'))
    model.save_weights(exp_name + '.pkl')
else:
    model.load_weights('experiments/baseline_test_.77-0.00.pkl')
    metrics = model.evaluate_generator(dataset.generator(batch_size, 'train'), nb_step)
    # print(metrics)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
