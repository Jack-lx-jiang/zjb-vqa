from keras import backend as K
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Masking, GRU, Input, Dropout, MaxPooling2D, RepeatVector, multiply, Softmax, Lambda, Reshape, \
    Conv1D, MaxPool1D, GlobalAveragePooling1D, Bidirectional, BatchNormalization
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model

from dataset import Dataset
from model.BaseModel import BaseModel
from util.loss import focal_loss
from util.metrics import multians_accuracy
from util.utils import load_embedding_weight


class ED_model(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'ED_model'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        # self.feature_dir = self.generate_feature_dir_name()
        self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
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
        video_dropout = Dropout(0.5)(video_reshape)
        decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(decoder)
        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(10), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class ShallowFeatureModel(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool', 'activation_40_maxpool2']
    model_name = 'ShallowFeatureModel'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = self.generate_feature_dir_name()
        # self.feature_dir = data_dir+'/ffinter15'
        BaseModel.__init__(self)

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
        video = Input((self.max_video_len, 1, 1, 2048))
        video_reshape = Reshape((self.max_video_len, 2048))(video)
        # video_reshape = Dropout(0.5)(video_reshape)
        shallow_feature_input = Input((self.max_video_len, 7, 7, 1024))
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)
        question_encoding = Bidirectional(GRU(1024))(Masking()(embedding_layer))
        # question_encoding = Lambda(lambda x: K.sum(x, axis=-2))(embedding_layer)

        # frame attention
        question_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(question_encoding)
        question_for_frame = RepeatVector(self.max_video_len)(question_part)

        video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(video_reshape)

        attention_score = Dense(1, kernel_regularizer=regularizers.l2(0.01))(multiply([question_for_frame, video_part]))
        attention = Softmax(axis=-2)(attention_score)
        video_joint = Lambda(lambda x: K.sum(x, axis=-2), name='video_joint')(multiply([video_reshape, attention]))

        # shallow attention
        shallow_feature = Reshape((self.max_video_len, 49, 1024))(shallow_feature_input)
        # shallow_feature = Dropout(0.5)(shallow_feature)
        region_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(shallow_feature)
        question_regin_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(question_encoding)
        question_for_region = RepeatVector(self.max_video_len * 49)(question_regin_part)
        question_for_region = Reshape((self.max_video_len, 49, -1))(question_for_region)
        region_attention_score = Dense(1, kernel_regularizer=regularizers.l2(0.01))(
            multiply([region_part, question_for_region]))
        region_attention = Softmax(axis=-2)(region_attention_score)
        shallow_joints = Lambda(lambda x: K.sum(x, axis=-2))(multiply([shallow_feature, region_attention]))

        shallow_joint_combine = Lambda(lambda x: K.sum(x, axis=-2), name='shallow_joint_sum')(
            multiply([shallow_joints, attention]))
        video_joint_sum = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(video_joint)
        shallow_joint_combine_sum = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(shallow_joint_combine)
        visual_feature_joint = multiply([video_joint_sum, shallow_joint_combine_sum])
        question_vq_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(question_encoding)
        vq_joint = multiply([visual_feature_joint, question_vq_part])
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(vq_joint)
        model = Model(inputs=[video, shallow_feature_input, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class FeatureConvModel(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool']
    model_name = 'FeatureConvModel'
    dataset = Dataset()

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.feature_dir = 'dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15'
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
        # question_encoding = Bidirectional(GRU(1024))(Masking()(embedding_layer))
        # question_encoding = Bidirectional(GRU(2048))(Masking()(embedding_layer))
        question_encoding = Bidirectional(GRU(512))(Masking()(embedding_layer))
        # video_dropout = Dropout(0.5)(video)
        video_gru = Bidirectional(GRU(1024, return_sequences=True))(video_reshape, initial_state=[question_encoding,
                                                                                                  question_encoding])
        # video_gru = Bidirectional(GRU(1024, return_sequences=True,kernel_regularizer=regularizers.l2(0.01)))(video_reshape, initial_state=question_encoding)
        video_conv1 = Conv1D(1024, 7, strides=2, activation='relu', kernel_regularizer=regularizers.l2(0.01))(video_gru)
        # video_conv1 = Conv1D(1024, 7, strides=2, activation='relu',kernel_regularizer=regularizers.l2(0.01))(video_reshape)
        video_bn1 = BatchNormalization()(video_conv1)
        video_pool1 = MaxPool1D(pool_size=3, strides=2)(video_bn1)
        video_conv2 = Conv1D(2048, 2, strides=1, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(video_pool1)
        video_bn2 = BatchNormalization()(video_conv2)
        video_conv3 = Conv1D(2048, 2, strides=1, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(video_bn2)
        video_bn3 = BatchNormalization()(video_conv3)
        video_pool2 = MaxPool1D()(video_bn3)

        video_conv4 = Conv1D(4096, 2, strides=1, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(video_pool2)
        video_bn4 = BatchNormalization()(video_conv4)
        video_conv5 = Conv1D(4096, 2, strides=1, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(video_bn4)
        video_bn5 = BatchNormalization()(video_conv5)
        video_conv6 = Conv1D(4096, 1, strides=1, activation='relu', padding='same',
                             kernel_regularizer=regularizers.l2(0.01))(video_bn5)

        video_avg_pool = GlobalAveragePooling1D()(video_conv6)

        question_part = Dense(4096, kernel_regularizer=regularizers.l2(0.01))(question_encoding)
        # question_part_bn = BatchNormalization()(question_part)
        video_part = Dense(4096, kernel_regularizer=regularizers.l2(0.01))(video_avg_pool)
        # video_part_bn = BatchNormalization()(video_part)
        # vq_joint = multiply([video_part_bn, question_part_bn])
        vq_joint = multiply([video_part, question_part])

        # logit = Dense(self.dataset.answer_size, activation='sigmoid')(video_avg_pool)
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(vq_joint)
        # logit = Dense(self.dataset.answer_size)(vq_joint)
        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(0.1), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model
