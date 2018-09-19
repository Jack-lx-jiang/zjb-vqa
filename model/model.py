from keras import backend as K
from keras.layers import Masking, GRU, RepeatVector, Concatenate, Softmax, multiply, Lambda, Add, Activation, Input, \
    Dropout
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
import numpy as np


def base_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')

    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len,
                                mask_zero=True)(question)
    question_encoding = GRU(300)(Masking()(embedding_layer))
    qe_repeat = RepeatVector(max_video_len)(question_encoding)
    video_question = Concatenate()([video, qe_repeat])
    attention_dense1 = Dense(512)(video_question)
    attention_dense2 = Dense(1)(attention_dense1)
    attention_score = Softmax(axis=-2)(attention_dense2)
    video_encoding = Lambda(lambda x: K.sum(x, axis=-2))(multiply([video, attention_score]))
    video_encoding2 = Dense(512)(video_encoding)
    question_encoding2 = Dense(512)(question_encoding)
    combine_encoding = multiply([video_encoding2, question_encoding2])
    decode = Dense(2048)(combine_encoding)
    logit = Dense(answer_size, activation='sigmoid')(decode)
    return Model(inputs=[video, question], outputs=logit)


def stacked_attention_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')

    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len,
                                mask_zero=True)(question)
    question_encoding = GRU(512)(Masking()(embedding_layer))
    question_dense1 = Dense(512)(question_encoding)
    video_dense1 = Dense(1024)(video)
    video_dense2 = Dense(512)(video_dense1)

    video_question = Activation('tanh')(Add()([video_dense2, question_dense1]))
    attention_dense1 = Dense(512)(video_question)
    attention_dense2 = Dense(1)(attention_dense1)
    attention_score = Softmax(axis=-2)(attention_dense2)

    video_encoding = Lambda(lambda x: K.sum(x, axis=-2))(multiply([video_dense2, attention_score]))
    # second reasoning period
    sr_question = Add()([video_encoding, question_encoding])
    # print(video_encoding.shape)
    # print(question_encoding.shape)
    # assert (video_encoding.shape == question_encoding.shape)
    sr_question_dense1 = Dense(512)(sr_question)
    sr_video_dense1 = Dense(1024)(video)
    sr_video_dense2 = Dense(512)(sr_video_dense1)
    sr_video_question = Activation('tanh')(Add()([sr_video_dense2, sr_question_dense1]))
    sr_attention_dense1 = Dense(512)(sr_video_question)
    sr_attention_dense2 = Dense(1)(sr_attention_dense1)
    sr_attention_score = Softmax(axis=-2)(sr_attention_dense2)

    video_attented = Lambda(lambda x: K.sum(x, axis=-2))(multiply([video_dense2, sr_attention_score]))
    final_video_question = Add()([video_attented, sr_question])

    decode = Dense(2048)(final_video_question)
    logit = Dense(answer_size, activation='sigmoid')(decode)
    return Model(inputs=[video, question], outputs=logit)


def encode_decode_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')
    embedding_matrix = np.load('embedding.npy')
    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len, mask_zero=True,
                                weights=[embedding_matrix], trainable=False)(question)
    question_encoding = GRU(512)(Masking()(embedding_layer))
    video_dropout = Dropout(0.5)(video)
    decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
    logit = Dense(answer_size, activation='sigmoid')(decoder)
    return Model(inputs=[video, question], outputs=logit)
