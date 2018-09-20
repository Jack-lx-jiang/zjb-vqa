import unittest

from keras.optimizers import Adadelta

from dataset import Dataset
from util.loss import focal_loss
from util.metrics import multians_accuracy


class TrainModelTest(unittest.TestCase):
    def test_base_model(self):
        data = Dataset(phase='train', base_dir='test/sample_data')
        views = __import__('model')
        cur_model = getattr(views, 'base_model')
        model = cur_model(data.vocabulary_size, data.max_question_len, data.max_video_len, data.frame_size,
                          data.answer_size, data.tokenizer)
        model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])
        model.fit_generator(data.generator(128, 'train'), 10)
        result = model.predict_generator(data.generator(128, 'train'), 10)
