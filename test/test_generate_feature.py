import itertools
import random
import unittest

from dataset import Dataset


class FeatureGenerationTest(unittest.TestCase):
    def test_generate_train_dataset_feature(self):
        data = Dataset(phase='train', base_dir='test/sample_data')
        data.phases = ['train']
        data.compute_frame_feature()
        batch_size = 128
        for x, y in itertools.islice(data.generator(batch_size, 'train'), random.randint(5, 20)):
            self.assertEqual(x[1].shape, (batch_size, data.max_question_len))
            self.assertEqual(x[0].shape, (batch_size, data.max_video_len, data.frame_size))
            self.assertEqual(y.shape, (batch_size, data.answer_size))
            total = y.sum()
            self.assertLessEqual(total, batch_size * 3)
            self.assertGreaterEqual(total, batch_size)
