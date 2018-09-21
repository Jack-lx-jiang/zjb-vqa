from keras import backend as K
from keras import regularizers
from keras.layers import Masking, GRU, RepeatVector, Concatenate, Softmax, multiply, Lambda, Add, Activation, Input, \
    Dropout, Conv2D, MaxPooling2D, Flatten
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model

from util.utils import load_embedding_weight
from util.utils import outer_product


def base_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
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


def stacked_attention_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
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


def encode_decode_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')
    embedding_matrix = load_embedding_weight(tokenizer)
    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len, mask_zero=True,
                                weights=[embedding_matrix], trainable=False)(question)
    question_encoding = GRU(512)(Masking()(Dropout(0.5)(embedding_layer)))
    video_dropout = Dropout(0.5)(video)
    decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
    logit = Dense(answer_size, activation='sigmoid')(decoder)
    return Model(inputs=[video, question], outputs=logit)


def bilinear_model(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')
    embedding_matrix = load_embedding_weight(tokenizer)
    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len, mask_zero=True,
                                weights=[embedding_matrix], trainable=False)(question)
    question_encoding = GRU(512)(Masking()(Dropout(0.5)(embedding_layer)))
    video_dropout = Dropout(0.5)(video)
    decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
    relevant_map = Lambda(outer_product, output_shape=(512, 512))([decoder, question_encoding])
    relevant_map = Lambda(lambda x: K.expand_dims(x))(relevant_map)
    conv1 = Conv2D(16, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(relevant_map)
    conv2 = Conv2D(16, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(
        conv1)
    conv3 = Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(conv2)
    conv4 = Conv2D(32, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(
        conv3)
    conv5 = Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(conv4)
    conv6 = Conv2D(64, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(
        conv5)
    conv7 = Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(conv6)
    conv8 = Conv2D(128, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(
        conv7)
    conv9 = Conv2D(256, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(conv8)
    conv10 = Conv2D(256, 3, strides=2, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(
        conv9)
    conv11 = Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(conv10)
    conv12 = Conv2D(512, 3, padding='same', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(conv11)
    pool1 = MaxPooling2D()(conv12)
    dense_input = Flatten()(pool1)
    dense1 = Dense(4096, kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(dense_input)
    dense2 = Dense(4096, kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(dense1)
    logit = Dense(answer_size, activation='sigmoid')(dense2)
    return Model(inputs=[video, question], outputs=logit)


def bilinear_model2(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')
    embedding_matrix = load_embedding_weight(tokenizer)
    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len, mask_zero=True,
                                weights=[embedding_matrix], trainable=False)(question)
    question_encoding = GRU(512)(Masking()(Dropout(0.5)(embedding_layer)))
    video_dropout = Dropout(0.5)(video)
    decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
    relevant_map = Lambda(outer_product, output_shape=(512, 512))([decoder, question_encoding])
    relevant_map = Lambda(lambda x: K.expand_dims(x))(relevant_map)
    conv1 = Conv2D(256, 512, padding='valid', kernel_regularizer=regularizers.l2(1.e-4), activation='relu')(
        relevant_map)
    # print(conv1.shape)
    dense_input = Flatten()(conv1)
    logit = Dense(answer_size, activation='sigmoid')(dense_input)
    return Model(inputs=[video, question], outputs=logit)


def bilinear_model3(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')
    embedding_matrix = load_embedding_weight(tokenizer)
    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len, mask_zero=True,
                                weights=[embedding_matrix], trainable=False)(question)
    question_encoding = GRU(512)(Masking()(Dropout(0.5)(embedding_layer)))
    video_dropout = Dropout(0.5)(video)
    video_encoding = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])

    question_part = Dense(1024)(question_encoding)
    video_part = Dense(1024)(video_encoding)
    joint = Dense(2048)(multiply([question_part, video_part]))
    logit = Dense(answer_size, activation='sigmoid')(joint)
    return Model(inputs=[video, question], outputs=logit)


def bilinear_model4(vocabulary_size, max_question_len, max_video_len, frame_size, answer_size, tokenizer):
    video = Input((max_video_len, frame_size))
    question = Input((max_question_len,), dtype='int32')
    embedding_matrix = load_embedding_weight(tokenizer)
    embedding_size = 300
    embedding_layer = Embedding(vocabulary_size, embedding_size, input_length=max_question_len, mask_zero=True,
                                weights=[embedding_matrix], trainable=False)(question)
    question_encoding = GRU(512)(Masking()(Dropout(0.5)(embedding_layer)))
    question_part = Dense(1024)(question_encoding)
    question_part = RepeatVector(max_video_len)(question_part)

    video_dropout = Dropout(0.5)(video)
    video_part = Dense(1024)(video_dropout)

    attention_score = Dense(1)(multiply([question_part, video_part]))
    attention = Softmax(axis=-2)(attention_score)
    video_encoding = Lambda(lambda x: K.sum(x, axis=-2))(multiply([video, attention]))

    question_part2 = Dense(1024)(question_encoding)
    video_part2 = Dense(1024)(video_encoding)
    joint = Dense(2048)(multiply([question_part2, video_part2]))
    logit = Dense(answer_size, activation='sigmoid')(joint)
    return Model(inputs=[video, question], outputs=logit)
