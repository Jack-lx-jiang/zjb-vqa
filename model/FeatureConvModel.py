from keras import regularizers
from keras.layers import GRU, Input, Reshape, \
    Conv1D, MaxPool1D, GlobalAveragePooling1D, Bidirectional, BatchNormalization
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adadelta
from keras.utils import multi_gpu_model

from dataset import Dataset
from model.BaseModel import BaseModel
from util.loss import focal_loss
from util.metrics import multians_accuracy


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

        # embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        # embedding_size = 300
        # embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size,
        #                             input_length=self.dataset.max_question_len, mask_zero=True,
        #                             weights=[embedding_matrix], trainable=False)(question)

        # question_encoding = Bidirectional(GRU(1024))(Masking()(embedding_layer))
        # question_encoding = Bidirectional(GRU(2048))(Masking()(embedding_layer))

        # question_encoding = Bidirectional(GRU(512))(Masking()(embedding_layer))

        # video_dropout = Dropout(0.5)(video)
        # video_gru = Bidirectional(GRU(1024, return_sequences=True))(video_reshape, initial_state=[question_encoding,
        #                                                                                           question_encoding])
        video_gru = Bidirectional(GRU(1024, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))(
            video_reshape)
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

        # question_part = Dense(4096, kernel_regularizer=regularizers.l2(0.01))(question_encoding)

        # question_part_bn = BatchNormalization()(question_part)
        # video_part = Dense(4096, kernel_regularizer=regularizers.l2(0.01))(video_avg_pool)
        # video_part_bn = BatchNormalization()(video_part)
        # vq_joint = multiply([video_part_bn, question_part_bn])

        # vq_joint = multiply([video_part, question_part])

        logit = Dense(self.dataset.answer_size, activation='sigmoid')(video_avg_pool)
        # logit = Dense(self.dataset.answer_size, activation='sigmoid')(vq_joint)
        # logit = Dense(self.dataset.answer_size)(vq_joint)
        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model = multi_gpu_model(model)
        # model.compile(optimizer=Adadelta(0.1), loss=binary_crossentropy, metrics=[multians_accuracy])
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model
