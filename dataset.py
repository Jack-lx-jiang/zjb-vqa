import math
import os
import random
from collections import Counter

import imageio
import keras
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from skimage import transform

from mapping import AnswerMapping


class Dataset():
    # phases = ['train', 'test', 'val']
    phases = ['train', 'test']
    BATCHES = 1000  # all batch
    MINI_BATCHES = 100  # frames pre batch

    def __init__(self, phase=None, base_dir=None, minimum_appear=3, interval=3):
        self.base_dir = base_dir or 'dataset'
        self.feature_dir = self.base_dir + '/feature'
        phase = phase or self.phases[0]

        self.dict = AnswerMapping()
        vid, questions, answers = self.preprocess_text(phase)
        # reduce the scale of answers
        # minium_count = 3

        rare_set = set([ans for ans, a_num in Counter(answers).items() if a_num <= minimum_appear])
        answers = [a for a in answers if a not in rare_set]
        ans = self.dict.tokenize(answers, True)
        self.answer_size = max(ans) + 1
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(questions)

        self.vocabulary_size = len(self.tokenizer.word_index) + 1
        self.max_video_len = 100
        self.max_question_len = 20
        self.interval = interval
        # the feature map size of each frame
        self.frame_size = 2048

    # split text data into three parts: video_id, questions and answers.
    def preprocess_text(self, phase):
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
    def generator(self, batch_size, phase, train_threshold=0.95, fluctuation=0.2, shallow_feature=""):
        if phase == 'val':
            vid, questions, answers = self.preprocess_text('train')
        else:
            vid, questions, answers = self.preprocess_text(phase)
        questions = self.tokenizer.texts_to_sequences(questions)
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
        all_index = [i for i in range(len(vid) * 5)]
        split = math.floor(len(vid) * train_threshold)
        if phase == 'train':
            inds = all_index[:split * 5]
        elif phase == 'val':
            inds = all_index[split * 5:]
        else:
            inds = all_index

        while True:
            if phase == 'train':
                random.shuffle(inds)
            count = 0
            while count < len(inds):
                X_video = np.zeros((batch_size, self.max_video_len, self.frame_size))
                if shallow_feature!= "":
                    X_shallow = np.zeros((batch_size, self.max_video_len, 14, 14, 1024))
                X_question = np.zeros((batch_size, self.max_question_len), dtype=np.int32)
                Y = np.zeros((batch_size, self.answer_size), dtype=np.int32)
                i = 0
                j = 0
                while i < batch_size:
                    try:
                        cur_question = inds[(count + j) % len(inds)]
                        # load feature maps of current question's video. shape: (video_len, feature_map_size)
                        cur_video = np.load(self.feature_dir + '/' + vid[cur_question // 5] + '_resnet.npy')
                        if shallow_feature != "":
                            cur_shallow = np.load(self.feature_dir+'_'+shallow_feature+ '/' + vid[cur_question // 5] + '_resnet.npy')
                            # print('extract')
                        # extract cur_video[:min{cur_video.shape[0],max_video_len}]
                        frames_num = min(self.max_video_len, math.ceil(cur_video.shape[0] / self.interval))
                        frame_indx = [f for f in range(0, frames_num * self.interval, self.interval)]
                        if self.interval >= 3 and fluctuation != 0.0:
                            changed_set = set()
                            for f in range(0, math.ceil(frames_num * fluctuation)):
                                cur_frame_indx = random.randint(0, frames_num - 1)
                                if cur_frame_indx not in changed_set:
                                    changed_set.add(cur_frame_indx)
                                    new_frame_indx = frame_indx[cur_frame_indx] + random.randint(-1, 1)
                                    if 0 <= new_frame_indx < cur_video.shape[0]:
                                        frame_indx[cur_frame_indx] = new_frame_indx
                        X_video[i, :frames_num] = cur_video[frame_indx]
                        X_shallow[i, :frames_num] = cur_shallow[frame_indx]
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
                    if shallow_feature != "":
                        yield [X_video, X_shallow, X_question], Y
                    else:
                        yield [X_video, X_question], Y
                except Exception as e:
                    print(str(e))
                count += j

    # extract video frames' feature
    def compute_frame_feature(self, target_layer='avg_pool'):
        origin_image_format = keras.backend.image_data_format()
        # set image_data_format to channels_last since some code is sensitive to channels
        keras.backend.set_image_data_format('channels_last')

        batch_size = self.MINI_BATCHES
        interval = 5
        res_model = ResNet50(weights='imagenet')
        model = Model(inputs=res_model.input, outputs=res_model.get_layer(target_layer).output)
        if target_layer == 'avg_pool':
            cur_feature_dir = self.feature_dir
        else:
            cur_feature_dir = self.feature_dir+'_'+target_layer
        for phase in self.phases:
            vid_dir = self.base_dir + '/' + phase
            for i, v in enumerate(os.listdir(vid_dir)):
                feature_file = cur_feature_dir + '/' + v.split('.')[0] + '_resnet.npy'
                if os.path.exists(feature_file):
                    continue
                print(vid_dir + '/' + str(v), i)
                video = imageio.get_reader(vid_dir + '/' + str(v), 'ffmpeg')
                if target_layer=='avg_pool':
                    vid_descriptors = np.zeros((self.BATCHES * batch_size, 2048))
                else:
                    vid_descriptors = np.zeros((self.BATCHES * batch_size,)+res_model.get_layer(target_layer).output_shape[1:])
                frame_count = 0
                frame_ind = 0
                stop = False
                for b in range(self.BATCHES):
                    batch = np.zeros((batch_size, 224, 224, 3))
                    for t in range(batch_size):
                        try:
                            frame = video.get_data(frame_ind)
                        except (imageio.core.CannotReadFrameError, IndexError):
                            stop = True
                            break
                        else:
                            frame = transform.resize(frame, (224, 224))
                            batch[t] = frame
                            frame_count += 1

                        frame_ind += interval
                    batch = preprocess_input(batch)
                    if target_layer == 'avg_pool':
                        vid_descriptors[b * batch_size:(b + 1) * batch_size] = model.predict_on_batch(batch).reshape(
                            batch.shape[0], -1)
                    else:
                        vid_descriptors[b * batch_size:(b + 1) * batch_size] = model.predict_on_batch(batch)
                    if stop:
                        break
                video.close()
                if not os.path.exists(cur_feature_dir):
                    os.mkdir(cur_feature_dir)
                np.save(feature_file, vid_descriptors[:frame_count])
        keras.backend.set_image_data_format(origin_image_format)