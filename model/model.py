from keras import backend as K
from keras.layers import Masking, GRU, RepeatVector, Concatenate, Softmax, multiply, Lambda
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding


def base_model(video, question, vocabulary_size, max_question_len, max_video_len, answer_size):
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
    return logit
