import os

from keras import backend as K
from keras import regularizers
from keras.layers import Masking, Input, Reshape, multiply, Bidirectional, Lambda, Softmax, Flatten, Dropout, \
    BatchNormalization, Activation, LSTM
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta

from dataset import Dataset
from layer.VladPooling import VladPooling
from model.BaseModel import BaseModel
from util.Kmeans import calculate_cluster_centers
from util.loss import focal_loss_with_entropy
from util.metrics import multians_accuracy
from util.utils import load_embedding_weight


# A trial Model that first train the language then the vision
# Unfortunately it get worse performance

class AssembleModel(BaseModel):
    minimum_appear = 20
    max_video_len = 100
    max_question_len = 20
    train_threshold = 0.95
    interval = 15
    nb_centers = 64
    features = ['avg_pool']
    model_name = 'AssembleModel'
    dataset = Dataset()
    language_train = False
    language_model = 'experiments/AssembleModel/2018-10-27_07-01-21/E170-L1.01-A0.43.pkl'

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

    def language_build(self):
        video = Input((self.max_video_len, 1, 1, 2048))
        # video_reshape = Reshape((self.max_video_len, 2048))(video)
        question = Input((self.max_question_len,), dtype='int32')
        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size, input_length=self.max_question_len,
                                    mask_zero=True, weights=[embedding_matrix], trainable=False)(question)

        question_encoding = Bidirectional(LSTM(512, return_sequences=True, recurrent_dropout=0.8),
                                          trainable=self.language_train)(Masking()(embedding_layer))
        question_encoding2 = Bidirectional(LSTM(512, recurrent_dropout=0.8), trainable=self.language_train)(
            question_encoding)

        logit = Dense(self.dataset.answer_size, activation='sigmoid', kernel_regularizer=regularizers.l2(0.001),
                      trainable=self.language_train)(question_encoding2)

        model = Model(inputs=[video, question], outputs=logit)
        model.summary()
        model.compile(optimizer=Adadelta(),
                      loss=[focal_loss_with_entropy(alpha=.25, gamma=2, ans_entropy=self.dataset.ans_entropy)],
                      metrics=[multians_accuracy])
        return model, video, question

    def build(self):
        if self.language_train:
            return self.language_build()[0]
        else:
            q_model, video, question = self.language_build()
            question_encoding2 = q_model.get_layer('bidirectional_2').output

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
            attention_score = Dense(2, kernel_regularizer=regularizers.l2(0.001))(
                multiply([visual_attention, video_part]))
            # attention_score = Dense(2, kernel_regularizer=regularizers.l2(0.001))(Activation('tanh')(multiply([visual_attention, video_part])))
            # attention_score = Dropout(0.5)(attention_score)
            attention = Softmax(axis=-2)(attention_score)
            attention = Lambda(lambda x: K.expand_dims(x, -1))(attention)
            video_reshape = Lambda(lambda x: K.expand_dims(x, -2))(pooled_bn)
            # video_joint = Lambda(lambda x: K.sum(x, axis=-2), name='video_joint')(multiply([pooled_bn, attention]))
            video_joint = Lambda(lambda x: K.sum(x, axis=-3), name='video_joint')(multiply([video_reshape, attention]))
            video_joint = Flatten()(video_joint)

            video_out = Dense(self.dataset.answer_size, kernel_regularizer=regularizers.l2(0.001))(video_joint)
            logit = Dense(self.dataset.answer_size, kernel_regularizer=regularizers.l2(0.001), activation='sigmoid')(
                multiply([video_out, q_model.output]))

            model = Model(inputs=[video, question], outputs=logit)
            model.summary()
            # model = multi_gpu_model(model)
            # model.compile(optimizer=Adadelta(0.1), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
            # model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
            # model.compile(optimizer=Adadelta(), loss=[focal_loss_weighted(alpha=.25, gamma=2, reverse_ans=self.dataset.reverse_ans, bias=17)], metrics=[multians_accuracy])
            model.compile(optimizer=Adadelta(),
                          loss=[focal_loss_with_entropy(alpha=.25, gamma=2, ans_entropy=self.dataset.ans_entropy)],
                          metrics=[multians_accuracy])

            model.load_weights(self.language_model, by_name=True)
            return model
