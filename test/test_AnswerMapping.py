import os
import random
import unittest

from faker import Faker

from mapping import AnswerMapping


class AnswerMappingTest(unittest.TestCase):
    def setUp(self):
        self.fake = Faker()

    def test_empty_init(self):
        mapping = AnswerMapping()
        self.assertEqual(mapping.ans2idx, {})
        self.assertEqual(mapping.idx2ans, [])

    def test_length(self):
        length = random.randint(3, 10)
        lst = self.fake.sentences(length)
        dt = {i: x for i, x in enumerate(lst)}
        mapping = AnswerMapping(dt, lst)
        self.assertEqual(mapping.ans2idx, dt)
        self.assertEqual(mapping.idx2ans, lst)
        self.assertEqual(len(mapping), len(dt))
        self.assertEqual(len(mapping), len(lst))

    def test_tokenize(self):
        length = random.randint(3, 10)
        token_list = self.fake.sentences(length * 2)
        token_list_1 = token_list[:length]
        token_list_2 = token_list[length:]
        mapping = AnswerMapping()
        mapping.tokenize(token_list_1, add_token=True)
        self.assertEqual(len(mapping), len(token_list_1))
        self.assertEqual(len(set(mapping.tokenize(token_list_1))), len(token_list_1))
        mapping.tokenize(token_list_2, add_token=True)
        self.assertEqual(len(mapping), len(token_list))
        self.assertEqual(len(set(mapping.tokenize(token_list))), len(token_list))

    def test_serialize(self):
        length = random.randint(3, 10)
        token_list = self.fake.sentences(length)
        mapping = AnswerMapping()
        mapping.tokenize(token_list, add_token=True)
        path = self.fake.uuid4()
        mapping.dump_to_file(path)
        mapping = AnswerMapping.load_from_file(path)
        self.assertEqual(len(mapping), len(token_list))
        self.assertEqual(len(set(mapping.tokenize(token_list))), len(token_list))
        os.remove(path)
