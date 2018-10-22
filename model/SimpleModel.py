import os

from keras import backend as K
from keras import regularizers
from keras.layers import Masking, GRU, Input, Reshape, MaxPool1D, multiply, Bidirectional, Lambda, TimeDistributed, \
    Softmax, Flatten, Concatenate, Dropout, BatchNormalization, Activation, LSTM
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


class MaxPoolModel(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'MaxPoolModel'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)
        video_pooled = MaxPool1D(pool_size=100)(video_reshape)
        video_pooled = Reshape((2048,))(video_pooled)

        question = Input((self.max_question_len,), dtype='int32')

        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size,
                                    input_length=self.dataset.max_question_len, mask_zero=True,
                                    weights=[embedding_matrix], trainable=False)(question)
        question_encoding = GRU(512)(Masking()(embedding_layer))

        # question_part = Dense(4096, kernel_regularizer=regularizers.l2(0.1))(question_encoding)
        # video_part = Dense(4096, kernel_regularizer=regularizers.l2(0.1))(video_pooled)
        # attention_score = Dense(2048, kernel_regularizer=regularizers.l2(0.1))(multiply([question_part, video_part]))

        # logit = Dense(self.dataset.answer_size, activation='sigmoid')(attention_score)

        ans_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)
        ans_p = Dense(self.dataset.answer_size, activation='sigmoid')(video_pooled)

        logit = multiply([ans_p, ans_mask])
        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class EncodeDecodeModel2(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'ED_model2'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
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
        question_encoding = Bidirectional(GRU(1024))(Masking()(embedding_layer))
        ans_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)

        # video_dropout = Dropout(0.5)(video_reshape)
        mask_video = Masking()(video_reshape)
        # decoder = Bidirectional(GRU(1024))(mask_video)

        each_frame_res = TimeDistributed(
            Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))(mask_video)
        mean_res = Lambda(lambda x: K.mean(x, axis=-2))(each_frame_res)

        # decoder = Lambda(lambda x: video_reshape[:,10,:])(video_reshape)
        # ans_p = Dense(self.dataset.answer_size, activation='sigmoid')(decoder)
        logit = multiply([mean_res, ans_mask])
        # model = Model(inputs=[video, question], outputs=logit)
        model = Model(inputs=[video, question], outputs=ans_mask)
        model.summary()
        # model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class AttentionModel(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'AttentionModel'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        question_encoding = Bidirectional(GRU(512))(Masking()(embedding_layer))

        question_encoding2 = Bidirectional(GRU(512))(Masking()(embedding_layer))

        # frame attention
        question_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(question_encoding)
        # question_for_frame = RepeatVector(self.max_video_len)(question_part)

        video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(video_reshape)

        attention_score = Dense(10, kernel_regularizer=regularizers.l2(0.01))(multiply([question_part, video_part]))
        attention = Softmax(axis=-2)(attention_score)
        attention = Reshape((self.max_video_len, 10, 1))(attention)
        video_reshape = Reshape((self.max_video_len, 1, 2048))(video_reshape)
        video_joint = Lambda(lambda x: K.sum(x, axis=-3), name='video_joint')(multiply([video_reshape, attention]))
        video_joint = Flatten()(video_joint)

        # ans_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)
        ans_p = Dense(self.dataset.answer_size, activation='sigmoid')(video_joint)

        # logit = multiply([ans_p, ans_mask])

        model = Model(inputs=[video, question], outputs=ans_p)
        # model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class AttentionModel2(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'AttentionModel2'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset_round2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        question_encoding = GRU(2048)(Masking()(embedding_layer))
        question_encoding = Reshape((1, 2048))(question_encoding)

        # frame attention
        # question_for_frame = RepeatVector(self.max_video_len)(question_encoding)
        video_mask = Masking()(video_reshape)
        attention_score = Lambda(lambda x: K.sum(x, axis=-1, keepdims=True))(multiply([question_encoding, video_mask]))
        attention = Softmax(axis=-2)(attention_score)

        # video_encoding = GRU(1024)(video_reshape)
        # video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01), activation='tanh')(video_encoding)
        #
        # attention_score = Dense(1, kernel_regularizer=regularizers.l2(0.01))(multiply([question_for_frame, video_part]))
        # attention = Softmax(axis=-2)(attention_score)
        each_frame_res = TimeDistributed(
            Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))(video_mask)

        # ans_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)
        ans_p = Lambda(lambda x: K.sum(x, axis=-2), name='video_joint')(multiply([each_frame_res, attention]))

        # logit = multiply([ans_p, ans_mask])

        model = Model(inputs=[video, question], outputs=ans_p)
        model.summary()
        # model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class CombineModel(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'CombineModel'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset_round2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        question_encoding = Bidirectional(GRU(512))(Masking()(embedding_layer))

        outputs = []
        for i in range(10):
            visual_attention = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(
                question_encoding)
            attented_video = multiply([visual_attention, video_reshape])
            video_output = GRU(512)(Masking()(attented_video))
            outputs.append(video_output)

        output_combine = Concatenate()(outputs)

        logit = Dense(self.dataset.answer_size, activation='sigmoid')(output_combine)

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class VladModel2(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 32
    features = ['activation_49']
    model_name = 'VladModel2'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset_round2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        self.kmeans_dir = self.feature_dir + '/kmeans_' + str(self.nb_centers) + '_' + self.features[0] + '.npy'
        BaseModel.__init__(self)
        if not os.path.exists(self.kmeans_dir):
            calculate_cluster_centers(self.feature_dir, self.features[0], self.nb_centers, 100, self.kmeans_dir)

    def build(self):
        video = Input((self.max_video_len, 7, 7, 2048))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        question_encoding = Bidirectional(GRU(512, return_sequences=True))(Masking()(embedding_layer))
        question_encoding2 = Bidirectional(GRU(512))(question_encoding)

        visual_attention = Dense(self.nb_centers * 128, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(
            question_encoding2)

        # pooled_feature = VladPooling(self.kmeans_dir, regularizer=regularizers.l2(0.00000000000004))(
        #     Masking()(video))
        pooled_feature = VladPooling(self.kmeans_dir)(Masking()(video))

        pooled_feature = Dropout(0.8)(pooled_feature)

        pooled_feature = Reshape((self.nb_centers, 2048))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)
        unstacked = Lambda(lambda x: K.tf.unstack(x, axis=1))(pooled_bn)
        sub_feature = [Dense(128)(x) for x in unstacked]
        sub_feature = Lambda(lambda x: K.concatenate(x))(sub_feature)
        sub_feature = multiply([sub_feature, visual_attention])
        ans_p = Dense(self.dataset.answer_size, kernel_regularizer=regularizers.l2(0.01))(sub_feature)

        # ans_mask = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(
        #     question_encoding2)
        #
        # logit = Activation('sigmoid')(multiply([ans_p, ans_mask]))
        logit = Activation('sigmoid')(ans_p)

        # logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(Concatenate()([sub_feature, question_encoding2]))

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class VladModel3(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 64
    # features = ['avg_pool']
    features = ['avg_pool']
    model_name = 'VladModel3'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        self.kmeans_dir = self.feature_dir + '/kmeans_' + str(self.nb_centers) + '_' + self.features[0] + '.npy'
        BaseModel.__init__(self)
        if not os.path.exists(self.kmeans_dir):
            calculate_cluster_centers(self.feature_dir, self.features[0], self.nb_centers, 100, self.kmeans_dir)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        # embd = K.variable(embedding_matrix)
        # mean = K.mean(embd, axis=0)
        # std = K.std(embd, axis=0)
        # embedding_layer = Lambda(lambda x: (x - mean) / std)(embedding_layer)

        question_encoding = Bidirectional(GRU(512, return_sequences=True))(Masking()(embedding_layer))
        question_encoding2 = Bidirectional(GRU(512))(question_encoding)

        visual_attention = Dense(self.nb_centers * 128, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(
            question_encoding2)
        #
        # visual_attention = Dense(1024, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(
        #     question_encoding2)
        # pooled_feature = VladPooling(self.kmeans_dir, regularizer=regularizers.l2(0.000000000000004))(
        #     Masking()(video))
        pooled_feature = VladPooling(self.kmeans_dir)(Masking()(video))

        pooled_feature = Dropout(0.8)(pooled_feature)

        pooled_feature = Reshape((self.nb_centers, 2048))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)
        unstacked = Lambda(lambda x: K.tf.unstack(x, axis=1))(pooled_bn)
        sub_feature = [Dense(128)(x) for x in unstacked]
        sub_feature = Lambda(lambda x: K.concatenate(x))(sub_feature)
        # sub_feature = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(pooled_bn)
        sub_feature = multiply([sub_feature, visual_attention])
        ans_p = Dense(self.dataset.answer_size, kernel_regularizer=regularizers.l2(0.01))(sub_feature)

        # ans_mask = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(
        #     question_encoding2)

        # logit = Activation('sigmoid')(multiply([ans_p, ans_mask]))
        logit = Activation('sigmoid')(ans_p)

        # logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(Concatenate()([sub_feature, question_encoding2]))

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class VladModel4(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 64
    features = ['avg_pool']
    model_name = 'VladModel4'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        self.kmeans_dir = self.feature_dir + '/kmeans_' + str(self.nb_centers) + '_' + self.features[0] + '.npy'
        BaseModel.__init__(self)
        if not os.path.exists(self.kmeans_dir):
            calculate_cluster_centers(self.feature_dir, self.features[0], self.nb_centers, 100, self.kmeans_dir)

    def build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)

        # embd = K.variable(embedding_matrix)
        # mean = K.mean(embd,axis=0)
        # std = K.std(embd, axis=0)
        # embedding_layer = Lambda(lambda x:(x-mean)/std)(embedding_layer)
        question_encoding = Bidirectional(LSTM(512, return_sequences=True))(Masking()(embedding_layer))
        question_encoding2 = LSTM(1024)(question_encoding)

        # visual_attention = Dense(1024)(question_encoding2)
        visual_attention = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(question_encoding2)
        # visual_attention = Dropout(0.5)(visual_attention)

        video_mask = Masking()(video)
        video_mask = Activation('relu')(video_mask)
        pooled_feature = VladPooling(self.kmeans_dir)(video_mask)

        pooled_feature = Dropout(0.8)(pooled_feature)

        pooled_feature = Reshape((self.nb_centers, 2048))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)
        # video_part = Dense(1024)(pooled_bn)
        video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(pooled_bn)
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
        logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(
            video_joint)
        #
        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model
