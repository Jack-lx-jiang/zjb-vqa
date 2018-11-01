from keras import backend as K
from keras import regularizers
from keras.applications.resnet50 import ResNet50
from keras.layers import Masking, GRU, Input, MaxPooling2D, RepeatVector, multiply, Softmax, Lambda, Reshape, \
    Bidirectional
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model

from dataset import Dataset
from model.BaseModel import BaseModel
from util.loss import focal_loss_sum
from util.metrics import multians_accuracy
from util.utils import load_embedding_weight


# There contains two model
# First one is a naive combination of shallow feature and high level feature
# Second one is a refined version, using the high level feature and language combination to guide
# the shallow feature attention

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
        # model.compile(optimizer=Adadelta(0.1),
        #               loss=[focal_loss_sum_with_entropy(alpha=.25, gamma=2, ans_entropy=self.dataset.ans_entropy)],
        #               metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(0.1), loss=[focal_loss_sum(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model


class ShallowFeatureModel2(BaseModel):
    minimum_appear = 3
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    features = ['avg_pool', 'activation_40_maxpool2']
    model_name = 'ShallowFeatureModel2'
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

        video_joint_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(video_joint)
        question_video_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(question_encoding)
        vq_summary = multiply([video_joint_part, question_video_part])

        # shallow attention
        shallow_feature = Reshape((self.max_video_len, 49, 1024))(shallow_feature_input)
        # shallow_feature = Dropout(0.5)(shallow_feature)
        region_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(shallow_feature)
        question_regin_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(vq_summary)
        question_for_region = RepeatVector(self.max_video_len * 49)(question_regin_part)
        question_for_region = Reshape((self.max_video_len, 49, -1))(question_for_region)
        region_attention_score = Dense(1, kernel_regularizer=regularizers.l2(0.01))(
            multiply([region_part, question_for_region]))
        region_attention = Softmax(axis=-2)(region_attention_score)
        shallow_joints = Lambda(lambda x: K.sum(x, axis=-2))(multiply([shallow_feature, region_attention]))

        shallow_joint_combine = Lambda(lambda x: K.sum(x, axis=-2), name='shallow_joint_sum')(
            multiply([shallow_joints, attention]))

        visual_feature_joint = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(shallow_joint_combine)
        question_vq_part = Dense(1024, kernel_regularizer=regularizers.l2(0.01))(vq_summary)
        vq_joint = multiply([visual_feature_joint, question_vq_part])
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(vq_joint)
        model = Model(inputs=[video, shallow_feature_input, question], outputs=logit)
        model.summary()
        # model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(0.1),
        #               loss=[focal_loss_sum_with_entropy(alpha=.25, gamma=2, ans_entropy=self.dataset.ans_entropy)],
        #               metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(0.1), loss=[focal_loss_sum(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model