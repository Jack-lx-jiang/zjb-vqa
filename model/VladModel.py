
import os

from keras import backend as K
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Dropout, BatchNormalization, Activation, LSTM
from keras.layers import Masking, GRU, Input, Reshape, MaxPooling2D, multiply, Bidirectional, Lambda, Softmax, Flatten
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model

from dataset import Dataset
from layer.VladPooling import VladPooling
from model.BaseModel import BaseModel
from util.Kmeans import calculate_cluster_centers
from util.loss import focal_loss, focal_loss_weighted, focal_loss_sum
from util.metrics import multians_accuracy
from util.utils import load_embedding_weight


# A Vlad model which doesnt use question directly
# and trying to use shallow feature
class VladModel(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['activation_40_maxpool2']
    model_name = 'VladModel'
    dataset = Dataset()
    nb_centers = 64

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        self.kmeans_dir = self.feature_dir + '/kmeans_' + str(self.nb_centers) + '.npy'
        BaseModel.__init__(self)
        if not os.path.exists(self.kmeans_dir):
            calculate_cluster_centers(self.feature_dir, self.features[0], self.nb_centers, self.kmeans_dir, 100)

    def preprocess_video_model(self):
        res_model = ResNet50(weights='imagenet')
        high_feature = res_model.get_layer(self.features[0]).output
        shallow_feature = res_model.get_layer('activation_40').output
        shallow_feature = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(shallow_feature)
        print(high_feature.shape)
        print(shallow_feature.shape)
        pre_model = Model(inputs=res_model.input, outputs=[high_feature, shallow_feature])
        return pre_model

    def build(self):
        # video = Input((self.max_video_len, 1, 1, 2048))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        # video_reshape = Dropout(0.5)(video_reshape)
        shallow_feature_input = Input((self.max_video_len, 7, 7, 1024))
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        question_encoding = Bidirectional(GRU(512))(Masking()(embedding_layer))
        # q_mask = Dense(self.dataset.answer_size, activation='sigmoid')(question_encoding)
        q_mask = Dense(self.nb_centers, activation='tanh')(question_encoding)

        # pooled_feature = VladPooling(self.kmeans_dir, regularizer=regularizers.l2(0.0000000000004))(
        #     Masking()(shallow_feature_input))
        pooled_feature = VladPooling(self.kmeans_dir)(Masking()(shallow_feature_input))

        pooled_feature = Dropout(0.8)(pooled_feature)

        pooled_feature = Reshape((self.nb_centers, 1024))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)
        unstacked = Lambda(lambda x: K.tf.unstack(x, axis=1))(pooled_bn)
        sub_feature = [Dense(1)(x) for x in unstacked]
        sub_feature = Lambda(lambda x: K.concatenate(x))(sub_feature)
        sub_feature = multiply([sub_feature, q_mask])
        ans_p = Dense(self.dataset.answer_size, activation='sigmoid')(sub_feature)

        # pooled_bn = BatchNormalization()(pooled_feature)
        # ans_p = Dense(self.dataset.answer_size, activation='sigmoid')(pooled_bn)

        # logit = multiply([q_mask, ans_p])
        # model = Model(inputs=[video, shallow_feature_input, question], outputs=ans_p)
        model = Model(inputs=[shallow_feature_input, question], outputs=ans_p)
        # model = Model(inputs=[video, shallow_feature_input, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        # model.compile(optimizer=AdamAccumulate(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


# A Vlad model which doesnt use question directly
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

        # logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))
        # (Concatenate()([sub_feature, question_encoding2]))

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


# A Vlad Model for high level feature
class VladModel4(BaseModel):
    minimum_appear = 20
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 64
    features = ['avg_pool']
    model_name = 'VladModel4'
    dataset = Dataset()
    stage2 = False
    stage1_model = 'experiments/  VladModel4/2018-10-26_07-16-59/E135-L1.05-A0.46.pkl'

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset_round2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        # self.feature_dir = 'dataset2/feature_avg_pool_len100_inter15'
        self.kmeans_dir = self.feature_dir + '/kmeans_' + str(self.nb_centers) + '_' + self.features[0] + '.npy'
        BaseModel.__init__(self)
        if not os.path.exists(self.kmeans_dir):
            calculate_cluster_centers(self.feature_dir, self.features[0], self.nb_centers, 100, self.kmeans_dir)
        # self.dataset.sample_frames = True

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
        question_encoding = Bidirectional(LSTM(512, return_sequences=True, recurrent_dropout=0.8),
                                          trainable=not self.stage2)(Masking()(embedding_layer))
        # question_encoding = LSTM(512, return_sequences=True, recurrent_dropout=0.8)
        # question_encoding.cell.trainable=not self.stage2
        # question_encoding = Bidirectional(question_encoding)(Masking()(embedding_layer))
        question_encoding2 = Bidirectional(LSTM(512, recurrent_dropout=0.8), trainable=not self.stage2)(
            question_encoding)
        # question_encoding2 = LSTM(512, recurrent_dropout=0.8)
        # question_encoding2.cell.trainable = not self.stage2
        # question_encoding2 = Bidirectional(question_encoding2)(question_encoding)
        # question_encoding = Bidirectional(GRU(512, return_sequences=True))(Masking()(embedding_layer))
        # question_encoding2 = Bidirectional(GRU(512))(question_encoding)

        # visual_attention = Dense(1024)(question_encoding2)
        visual_attention = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(question_encoding2)
        visual_attention = Activation('tanh')(visual_attention)
        # visual_attention = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(question_encoding2)
        # visual_attention = Dropout(0.5)(visual_attention)

        video_mask = Masking()(video)
        # video_bn = BatchNormalization()(video_mask)
        # video_mask = Activation('relu')(video_mask)
        pooled_feature = VladPooling(self.kmeans_dir)(video_mask)

        pooled_feature = Dropout(0.8)(pooled_feature)

        pooled_feature = Reshape((self.nb_centers, 2048))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)
        # video_part = Dense(1024)(pooled_bn)
        video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.0001))(pooled_bn)
        video_part = Activation('tanh')(video_part)
        # video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.001))(pooled_bn)
        # video_part = BatchNormalization()(video_part)
        # video_part = Dropout(0.5)(video_part)

        # attention_score = Dense(1)(multiply([visual_attention, video_part]))
        attention_score = Dense(2, kernel_regularizer=regularizers.l2(0.001))(multiply([visual_attention, video_part]))
        # attention_score = Dense(2, kernel_regularizer=regularizers.l2(0.001))(Activation('tanh')
        # (multiply([visual_attention, video_part])))
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

        # question_encoding_for_ans
        # question_encoding_ans = Bidirectional(LSTM(512, dropout=0.8))(Masking()(embedding_layer))

        # combine part
        video_joint_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(video_joint)
        question_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.001),
                               trainable=not self.stage2)(question_encoding2)
        # question_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.001))
        # (question_encoding_ans)
        logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(
            multiply([video_joint_dense, question_dense]))
        #
        # combine part2
        # video_score = Dense(2048, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(
        #     video_joint)
        # question_score = Dense(2048, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))(
        #     question_encoding2)
        # logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001))
        # (multiply([video_score, question_score]))

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        # model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(),
                      loss=[focal_loss_weighted(alpha=.25, gamma=2, reverse_ans=self.dataset.ans_entropy, bias=17)],
                      metrics=[multians_accuracy])
        # model.compile(optimizer=Adadelta(),
        # loss=[focal_loss_with_entropy(alpha=.25, gamma=2, ans_entropy=self.dataset.ans_entropy)],
        #  metrics=[multians_accuracy])
        if self.stage2:
            model.load_weights(self.stage1_model)
        return model


# A trial to use Shallow Feature in Vlad Model
class VladModel5(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 32
    features = ['activation_40_maxpool2']
    model_name = 'VladModel5'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset_round2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
        # self.feature_dir = 'dataset2/feature_avg_pool_len100_inter15'
        self.kmeans_dir = self.feature_dir + '/kmeans_' + str(self.nb_centers) + '_' + self.features[0] + '.npy'
        BaseModel.__init__(self)
        if not os.path.exists(self.kmeans_dir):
            calculate_cluster_centers(self.feature_dir, self.features[0], self.nb_centers, 100, self.kmeans_dir)
        # self.dataset.sample_frames = True

    def build(self):
        video = Input((self.max_video_len, 7, 7, 1024))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)

        question_encoding = Bidirectional(LSTM(512, return_sequences=True, recurrent_dropout=0.8))(
            Masking()(embedding_layer))
        question_encoding2 = Bidirectional(LSTM(512, recurrent_dropout=0.8))(question_encoding)
        visual_attention = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(question_encoding2)

        video_mask = Masking()(video)
        pooled_feature = VladPooling(self.kmeans_dir)(video_mask)

        pooled_feature = Reshape((self.nb_centers, 1024))(pooled_feature)
        pooled_bn = BatchNormalization()(pooled_feature)

        pooled_bn = Dropout(0.8)(pooled_bn)
        video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(pooled_bn)

        # attention_score = Dense(1)(multiply([visual_attention, video_part]))
        attention_score = Dense(4, kernel_regularizer=regularizers.l2(0.01))(multiply([visual_attention, video_part]))

        attention = Softmax(axis=-2)(attention_score)
        attention = Lambda(lambda x: K.expand_dims(x, -1))(attention)
        video_reshape = Lambda(lambda x: K.expand_dims(x, -2))(pooled_bn)
        # video_joint = Lambda(lambda x: K.sum(x, axis=-2), name='video_joint')(multiply([pooled_bn, attention]))
        video_joint = Lambda(lambda x: K.sum(x, axis=-3), name='video_joint')(multiply([video_reshape, attention]))
        video_joint = Flatten()(video_joint)

        # combine part
        # video_joint_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.001))(video_joint)
        video_joint_dense = Dense(2048, kernel_regularizer=regularizers.l2(0.01))(video_joint)
        # video_joint_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(video_joint)
        # question_dense = Dense(2048, activation='tanh', kernel_regularizer=regularizers.l2(0.01))(question_encoding2)
        question_dense = Dense(2048, kernel_regularizer=regularizers.l2(0.01))(question_encoding2)
        logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01))(
            multiply([video_joint_dense, question_dense]))

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss_sum(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model
