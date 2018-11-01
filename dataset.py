import math
import random
from collections import Counter

import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from numpy.random import choice

from mapping import AnswerMapping


class Dataset():
    phases = ['train', 'test', 'val']

    def __init__(self):
        self.configured = False
        pass

    def set_config(self, base_dir=None, minimum_appear=3, max_video_len=100, max_question_len=20, train_threshold=0.95,
                   interval=1, feature_dir='dataset/feature', feature=['avg_pool'], sample_frames=False):
        self.configured = True
        self.base_dir = base_dir or 'dataset'
        self.max_video_len = max_video_len
        self.train_threshold = train_threshold
        self.interval = interval
        print('extract interval:' + str(interval))
        self.feature_dir = feature_dir
        self.feature = feature

        self.dict = AnswerMapping()
        vid, questions, answers = self.preprocess_text('train')

        # generate reverse answers vector
        ca = Counter(answers)

        rare_set = set([ans for ans, a_num in ca.items() if a_num <= minimum_appear])
        answers = [a for a in answers if a not in rare_set]
        # total_ans = sum(ca[a] for a in answers)
        # for a in answers:
        #     print(a,  ca[a])
        # print(total_ans)
        # print(sum(ca.values()))
        ans = self.dict.tokenize(answers, True)

        self.answer_size = max(ans) + 1

        total_ans = sum(ca[self.dict.idx2ans[i]] for i in range(self.answer_size))
        # print(total_ans)

        self.ans_entropy = np.asarray(
            [-math.e * math.log(ca[self.dict.idx2ans[i]] / total_ans) * ca[self.dict.idx2ans[i]] / total_ans for i in
             range(self.answer_size)])
        self.total_entropy = sum(self.ans_entropy)
        # print(self.total_entropy)
        self.ans_p = self.ans_entropy / self.total_entropy
        self.reverse_ans = np.asarray([1 / ca[self.dict.idx2ans[i]] for i in range(self.answer_size)])
        # self.reverse_ans = np.asarray([1/math.log(ca[self.dict.idx2ans[i]]) for i in range(self.answer_size)])
        # print(self.reverse_ans)

        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(questions)

        self.vocabulary_size = len(self.tokenizer.word_index) + 1
        self.max_question_len = max_question_len

        self.sample_frames = sample_frames
        print('finish dataset configuration')

    def get_nb_steps(self, phase, batch_size):
        assert self.configured
        if phase == 'val':
            vid, questions, answers = self.preprocess_text('train')
        else:
            vid, questions, answers = self.preprocess_text(phase)
        if phase == 'train':
            nb_samples = math.floor(len(vid) * 5 * self.train_threshold)
        elif phase == 'val':
            nb_samples = math.ceil(len(vid) * 5 * (1 - self.train_threshold))
        else:
            nb_samples = len(vid) * 5
        return nb_samples // batch_size + 1

    # split text data into three parts: video_id, questions and answers.
    def preprocess_text(self, phase):
        assert self.configured
        assert (phase in self.phases)

        vid = []
        questions = []
        answers = []
        fs = open(self.base_dir + '/' + phase + '.txt')
        for line in fs.readlines():
            parts = [x.strip() for x in line.split(',')]
            vid.append(parts[0])
            for i in range(5):
                questions.append(parts[i * 4 + 1])
                answers.extend(parts[i * 4 + 2:i * 4 + 5])
        fs.close()
        return vid, questions, answers

    # the generator function for model's input
    def generator(self, phase, batch_size):
        assert self.configured
        if phase == 'val':
            vid, questions, answers = self.preprocess_text('train')
        else:
            vid, questions, answers = self.preprocess_text(phase)
        questions = self.tokenizer.texts_to_sequences(questions)
        # transfer answers to one_hot format
        if phase != 'test':
            answers = self.dict.tokenize(answers)
            one_hot_answers = []
            for i in range(0, len(answers), 3):
                cur_answer = np.zeros(self.answer_size)
                cur_answer += to_categorical(answers[i], self.answer_size) if answers[i] != -1 else np.zeros(
                    self.answer_size)
                cur_answer += to_categorical(answers[i + 1], self.answer_size) if answers[i + 1] != -1 else np.zeros(
                    self.answer_size)
                cur_answer += to_categorical(answers[i + 2], self.answer_size) if answers[i + 2] != -1 else np.zeros(
                    self.answer_size)
                one_hot_answers.append(cur_answer)
            assert (len(one_hot_answers) == len(questions))

        # split dataset
        all_index = [i for i in range(len(vid) * 5)]
        split = math.floor(len(vid) * self.train_threshold)
        if phase == 'train':
            inds = all_index[:split * 5]
        elif phase == 'val':
            inds = all_index[split * 5:]
        else:
            inds = all_index

        # get outputs shape
        output_shape = [np.load(self.feature_dir + '/' + vid[0] + '_' + f + '_resnet.npy').shape[1:] for f in
                        self.feature]

        bs = batch_size

        # means = [np.load(self.feature_dir+'/'+f+'_mean.npy')[np.newaxis,:] for f in self.feature]
        # stds = [np.load(self.feature_dir+'/'+f+'_std.npy')[np.newaxis, :] for f in self.feature]
        while True:
            if phase == 'train':
                random.shuffle(inds)
            count = 0
            while count < len(inds):
                # start = time()
                X_videos = [np.zeros((bs, self.max_video_len,) + os) for os in output_shape]
                X_question = np.zeros((bs, self.max_question_len), dtype=np.int32)
                Y = np.zeros((bs, self.answer_size), dtype=np.int32)
                i = 0
                j = 0
                while i < bs:
                    try:
                        cur_question = inds[(count + j) % len(inds)]
                        for fi, f in enumerate(self.feature):
                            # load feature maps of current question's video. shape: (video_len, feature_map_size)
                            cur_video = np.load(
                                self.feature_dir + '/' + vid[cur_question // 5] + '_' + f + '_resnet.npy')
                            # cur_video -= means[fi]
                            # cur_video /= stds[fi]
                            # extract cur_video[:min{cur_video.shape[0],max_video_len}]
                            cur_video_len = math.ceil(cur_video.shape[0] / self.interval)
                            X_videos[fi][i, :cur_video_len] = cur_video[
                                                              :self.max_video_len * self.interval:self.interval]
                            if self.sample_frames and cur_video_len < self.max_video_len:
                                sample_indx = sorted(choice(range(cur_video_len), self.max_video_len))
                                X_videos[fi][i, :] = X_videos[fi][i, sample_indx]
                        q = questions[cur_question]
                        X_question[i, :len(q)] = q
                        if phase != 'test':
                            Y[i, :] = one_hot_answers[cur_question]
                        i += 1
                    except Exception as e:
                        print('generator error:')
                        print(str(e))
                    j += 1
                Y[Y > 1] = 1
                try:
                    yield X_videos + [X_question], Y
                except Exception as e:
                    print(str(e))
                count += j
                # stop = time()
                # print('   '+str(stop-start) + "s")
