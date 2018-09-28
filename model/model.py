from keras.layers import Masking, GRU, Input, \
    Dropout
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta

from dataset import Dataset
from util.loss import focal_loss
from util.metrics import multians_accuracy
from util.utils import load_embedding_weight

from model.BaseModel import BaseModel
from keras.layers import Masking, GRU, Input, \
    Dropout
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.optimizers import Adadelta

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
        self.feature_dir = self.generate_feature_dir_name()
        BaseModel.__init__(self)

    def build(self):
        video = Input((self.max_video_len, 2048))

        question = Input((self.max_question_len,), dtype='int32')

        embedding_matrix = load_embedding_weight(self.dataset.tokenizer)
        embedding_size = 300
        embedding_layer = Embedding(self.dataset.vocabulary_size, embedding_size,
                                    input_length=self.dataset.max_question_len, mask_zero=True,
                                    weights=[embedding_matrix], trainable=False)(question)
        question_encoding = GRU(512)(Masking()(embedding_layer))
        video_dropout = Dropout(0.5)(video)
        decoder = GRU(512)(Masking()(video_dropout), initial_state=[question_encoding])
        logit = Dense(self.dataset.answer_size, activation='sigmoid')(decoder)
        model = Model(inputs=[video, question], outputs=logit)
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        return model
