import os
import random

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
    vocabulary_size = 10000
    BATCHES = 1000  # all batch
    MINI_BATCHES = 100  # frames pre batch

    def __init__(self, phase=None, base_dir=None):
        self.base_dir = base_dir or 'dataset'
        self.feature_dir = self.base_dir + '/feature'
        phase = phase or self.phases[0]

        self.dict = AnswerMapping()
        vid, questions, answers = self.preprocess_text(phase)
        ans = self.dict.tokenize(answers, True)
        self.answer_size = max(ans) + 1
        self.tokenizer = Tokenizer(self.vocabulary_size)
        self.tokenizer.fit_on_texts(questions)

        self.max_video_len = 500
        self.max_question_len = 20
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
            parts = line.split(',')
            vid.append(parts[0])
            for i in range(5):
                questions.append(parts[i * 4 + 1])
                answers.extend(parts[i * 4 + 2:i * 4 + 5])
        fs.close()
        return vid, questions, answers

    # the generator function for model's input
    def generator(self, batch_size, phase):
        assert (phase in self.phases)

        vid, questions, answers = self.preprocess_text(phase)
        questions = self.tokenizer.texts_to_sequences(questions)
        if phase == 'train':
            answers = self.dict.tokenize(answers)
            one_hot_answers = [to_categorical(answers[i], self.answer_size) +
                               to_categorical(answers[i + 1], self.answer_size) +
                               to_categorical(answers[i + 2], self.answer_size)
                               for i in range(0, len(answers), 3)]
        inds = [i for i in range(len(vid) * 5)]
        assert (len(inds) == len(questions))
        if phase == 'train':
            assert (len(one_hot_answers) == len(questions))
        while True:
            if phase == 'train':
                random.shuffle(inds)
            count = 0
            while count < len(inds):
                X_video = np.zeros((batch_size, self.max_video_len, self.frame_size))
                X_question = np.zeros((batch_size, self.max_question_len), dtype=np.int32)
                Y = np.zeros((batch_size, self.answer_size), dtype=np.int32)
                i = 0
                j = 0
                while i < batch_size:
                    try:
                        cur_question = inds[(count + j) % len(inds)]
                        # load feature maps of current question's video. shape: (video_len, feature_map_size)
                        cur_video = np.load(self.feature_dir + '/' + vid[cur_question // 5] + '_resnet.npy')
                        # extract cur_video[:min{cur_video.shape[0],max_video_len}]
                        X_video[i, :cur_video.shape[0]] = cur_video[:self.max_video_len]
                        q = questions[cur_question]
                        X_question[i, :len(q)] = q
                        if phase == 'train':
                            Y[i, :] = one_hot_answers[cur_question]
                        i += 1
                    except Exception as e:
                        print('generator error:')
                        print(str(e))
                    j += 1
                Y[Y > 1] = 1
                try:
                    yield [X_video, X_question], Y
                except Exception as e:
                    print(str(e))
                count += j

    # extract video frames' feature
    def compute_frame_feature(self):
        origin_image_format = keras.backend.image_data_format()
        # set image_data_format to channels_last since some code is sensitive to channels
        keras.backend.set_image_data_format('channels_last')

        batch_size = self.MINI_BATCHES
        interval = 5
        res_model = ResNet50(weights='imagenet')
        model = Model(inputs=res_model.input, outputs=res_model.get_layer('avg_pool').output)

        for phase in self.phases:
            vid_dir = self.base_dir + '/' + phase
            for i, v in enumerate(os.listdir(vid_dir)):
                feature_file = self.feature_dir + '/' + v.split('.')[0] + '_resnet.npy'
                if os.path.exists(feature_file):
                    continue
                video = imageio.get_reader(vid_dir + '/' + str(v), 'ffmpeg')
                print(vid_dir + '/' + str(v), i)
                vid_descriptors = np.zeros((self.BATCHES * batch_size, 2048))
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
                    vid_descriptors[b * batch_size:(b + 1) * batch_size] = model.predict_on_batch(batch).reshape(
                        batch.shape[0], -1)
                    if stop:
                        break
                video.close()
                if not os.path.exists(self.feature_dir):
                    os.mkdir(self.feature_dir)
                np.save(feature_file, vid_descriptors[:frame_count])
        keras.backend.set_image_data_format(origin_image_format)
