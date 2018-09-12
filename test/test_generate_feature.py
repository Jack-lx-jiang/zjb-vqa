import unittest
from dataset import Dataset


class FeatureGenerationTest(unittest.TestCase):
    def test(self):
        dataset = Dataset(phase='train', base_dir='test/sample_data')
        dataset.phases = ['train']
        dataset.compute_frame_feature()
