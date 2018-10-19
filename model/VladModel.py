import os

from keras import backend as K
from keras.applications.resnet50 import ResNet50
from keras.layers import Masking, GRU, Input, MaxPooling2D, multiply, Lambda, Reshape, \
    Bidirectional, BatchNormalization, Dropout
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
