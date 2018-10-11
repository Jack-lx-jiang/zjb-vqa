import unittest
from dataset import Dataset


class TrainModelTest(unittest.TestCase):
    def test_base_model(self):
        data_dir = 'test/sample_data'
        data = Dataset()
        data.set_config(base_dir=data_dir)
        views = __import__('model')
        cur_model = getattr(views, 'EncodeDecodeModel')
        model = cur_model(data_dir)
        model.train(128, 10)
