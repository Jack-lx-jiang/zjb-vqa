import os

from keras import backend as K
from keras import regularizers
from keras.layers import Masking, GRU, Input, Reshape, multiply, Bidirectional, Lambda, Softmax, Flatten, Dropout, \
    BatchNormalization, LSTM, Add
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model

from dataset import Dataset
from layer.VladPooling import VladPooling
from model.BaseModel import BaseModel
from util.Kmeans import calculate_cluster_centers
from util.loss import focal_loss
from util.metrics import multians_accuracy
from util.utils import load_embedding_weight


class RGB_OPT_Model(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 64
    features = ['avg_pool', 'opt_avg_pool']
    model_name = 'RGB_OPT_Model'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset2/feature_avg_pool_len100_inter15'
        self.kmeans_dirs = [self.feature_dir + '/kmeans_' + str(self.nb_centers) + '_' + f + '.npy' for f in
                            self.features]
        BaseModel.__init__(self)
        for i in range(len(self.kmeans_dirs)):
            if not os.path.exists(self.kmeans_dirs[i]):
                print(self.kmeans_dirs[i])
                calculate_cluster_centers(self.feature_dir, self.features[i], self.nb_centers, 100, self.kmeans_dirs[i])

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        opt = Input((self.max_video_len, 1, 1, 2048))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)

        question_encoding = Bidirectional(LSTM(512, return_sequences=True))(Masking()(embedding_layer))
        question_encoding2 = Bidirectional(LSTM(512))(question_encoding)
        # question_encoding = Bidirectional(GRU(512, return_sequences=True))(Masking()(embedding_layer))
        # question_encoding2 = Bidirectional(GRU(512))(question_encoding)

        # visual_attention = Dense(1024)(question_encoding2)
        visual_attention = Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='tanh')(question_encoding2)
        # visual_attention = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(question_encoding2)
        # visual_attention = Dropout(0.5)(visual_attention)

        opt_mask = Masking()(opt)
        pooled_opt = VladPooling(self.kmeans_dirs[1])(opt_mask)
        pooled_opt = Dropout(0.8)(pooled_opt)
        pooled_opt = Reshape((self.nb_centers, 2048))(pooled_opt)
        pooled_opt_bn = BatchNormalization()(pooled_opt)

        video_mask = Masking()(video)
        # video_mask = Activation('relu')(video_mask)
        pooled_feature = VladPooling(self.kmeans_dirs[0])(video_mask)

        pooled_feature = Dropout(0.8)(pooled_feature)

        pooled_feature = Reshape((self.nb_centers, 2048))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)
        # video_part = Dense(1024)(pooled_bn)

        later_fusion = Add()([pooled_bn, pooled_opt_bn])
        video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.0001), activation='tanh')(later_fusion)
        # video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(pooled_bn)
        # video_part = BatchNormalization()(video_part)
        # video_part = Dropout(0.5)(video_part)

        # attention_score = Dense(1)(multiply([visual_attention, video_part]))
        attention_score = Dense(2, kernel_regularizer=regularizers.l2(0.001))(multiply([visual_attention, video_part]))
        # attention_score = Dense(2, kernel_regularizer=regularizers.l2(0.001))(Activation('tanh')(multiply([visual_attention, video_part])))
        # attention_score = Dropout(0.5)(attention_score)
        attention = Softmax(axis=-2)(attention_score)
        attention = Lambda(lambda x: K.expand_dims(x, -1))(attention)
        video_reshape = Lambda(lambda x: K.expand_dims(x, -2))(pooled_bn)
        # video_joint = Lambda(lambda x: K.sum(x, axis=-2), name='video_joint')(multiply([pooled_bn, attention]))
        video_joint = Lambda(lambda x: K.sum(x, axis=-3), name='video_joint')(multiply([video_reshape, attention]))
        video_joint = Flatten()(video_joint)

        # video_joint = Dropout(0.5)(video_joint)

        # logit = Dense(self.dataset.answer_size, activation='sigmoid')(video_joint)
        # logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(
        #     video_joint)

        # combine part
        video_joint_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(video_joint)
        question_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(question_encoding2)
        logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(
            multiply([video_joint_dense, question_dense]))

        model = Model(inputs=[video, opt, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class EncodeDecode_opt_Model(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['opt_avg_pool']
    model_name = 'ED_opt_model'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset2/feature_avg_pool_len100_inter15'
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)

        question = Input((self.max_question_len,), dtype='int32')

        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size,
                                    input_length=self.dataset.max_question_len, mask_zero=True,
                                    weights=[embedding_matrix], trainable=False)(question)
        question_encoding = GRU(512)(Masking()(embedding_layer))
        ans_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)
        video_dropout = Dropout(0.5)(Masking()(video_reshape))
        decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(decoder)
        model = Model(inputs=[video, question], outputs=logit)
        # model = Model(inputs=[video, question], outputs=ans_mask)
        model.summary()
        try:
            model = multi_gpu_model(model)
        except ValueError:
            pass
        # model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        # model.compile(optimizer=Adadelta(), loss=[focal_loss_mean(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class EncodeDecode_RGB_opt_Model(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool', 'opt_avg_pool']
    model_name = 'ED_RGB_opt_model'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset2/feature_avg_pool_len100_inter15'
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)

        opt = Input((self.max_video_len, 1, 1, 2048))
        opt_reshape = Reshape((self.max_video_len, 2048))(opt)

        question = Input((self.max_question_len,), dtype='int32')

        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size,
                                    input_length=self.dataset.max_question_len, mask_zero=True,
                                    weights=[embedding_matrix], trainable=False)(question)
        question_encoding = GRU(512, recurrent_dropout=0.8)(Masking()(embedding_layer))
        ans_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)
        # video_dropout = Dropout(0.5)(Masking()(video_reshape))
        decoder = GRU(512, recurrent_dropout=0.8)(Masking()(video_reshape), initial_state=[question_encoding])
        decoder2 = GRU(512, recurrent_dropout=0.8)(Masking()(opt_reshape), initial_state=[question_encoding])
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(Add()([decoder, decoder2]))
        model = Model(inputs=[video, opt, question], outputs=logit)
        # model = Model(inputs=[video, question], outputs=ans_mask)
        model.summary()
        try:
            model = multi_gpu_model(model)
        except ValueError:
            pass
        # model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        # model.compile(optimizer=Adadelta(), loss=[focal_loss_mean(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model
