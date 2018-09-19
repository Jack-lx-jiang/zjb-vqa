import hashlib
import unittest

from dataset import Dataset


class WordEmbeddingTest(unittest.TestCase):
    def test_fix_question_tokenizer(self):
        data = Dataset(phase='train', base_dir='test/sample_data')
        tokenizer = data.tokenizer
        all_token = ""
        for token in tokenizer.word_index:
            all_token += token
        sha256 = hashlib.sha256(all_token.encode('utf-8')).hexdigest()

        # this ensures that index in tokenizer will not change in different sessions
        self.assertEqual(sha256, '6c076edeafd2203e7ae9d20afccc6b76809cb03faf06452695ad59820656de06')
