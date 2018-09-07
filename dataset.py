import random, numpy as np
import imageio
import skimage
from skimage import transform
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.models import Model
import pickle as p
import os
from mapping import AnswerMapping


class Dataset():
    # phases = ['train', 'test', 'val']
    phases = ['train', 'test']
    base_dir = 'dataset'
    vocabulary_size = 10000
    feature_dir = base_dir + '/feature'

    def __init__(self):
        self.dict = AnswerMapping()
        vid, questions, answers = self.preprocess_text(self.phases[0])
        ans = self.dict.tokenize(answers, True)
        self.answer_size = max(ans) + 1
        self.tokenizer = Tokenizer(self.vocabulary_size)
        self.tokenizer.fit_on_texts(questions)

        self.max_video_len = 100
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
        return vid, questions, answers

    # the generator function for model's input
    def generator(self, batch_size, phase):
        assert (phase in self.phases)

        vid, questions, answers = self.preprocess_text(phase)
        questions = self.tokenizer.texts_to_sequences(questions)
        answers = self.dict.tokenize(answers)
        one_hot_answers = [to_categorical(answers[i], self.answer_size) + \
                           to_categorical(answers[i + 1], self.answer_size) + \
                           to_categorical(answers[i + 2], self.answer_size) \
                           for i in range(0, len(answers), 3)]

        inds = [i for i in range(len(vid) * 5)]
        assert (len(inds) == len(questions))
        assert (len(one_hot_answers) == len(questions))
        while True:
            if phase == 'train':
                random.shuffle(inds)
            count = 0
            while (count < len(inds)):
                X_video = np.zeros((batch_size, self.max_video_len, self.frame_size))
                X_question = np.zeros((batch_size, self.max_question_len), dtype=np.int32)
                Y = np.zeros((batch_size, self.answer_size), dtype=np.int32)
                i = 0
                j = 0
                while i < batch_size:
                    try:
                        # load feature map of each frame
                        cur_video = np.load(self.feature_dir + '/' + vid[(count + j) % len(inds) // 5] + '_resnet.npy')
                        X_video[i, :cur_video.shape[0]] = cur_video[:self.max_video_len]
                        q = questions[inds[(count + j) % len(inds)]]
                        X_question[i, :len(q)] = q
                        Y[i, :] = one_hot_answers[(count + j) % len(inds)]
                        i += 1
                    except Exception as e:
                        print('generator error:')
                        print(str(e))
                j += 1
            Y[Y > 1] = 1
            yield [X_video, X_question], Y
            count += j

    # extract video frames' feature
    def compute_frame_feature(self):
        batch_size = 100
        interval = 5
        res_model = ResNet50(weights='imagenet')
        model = Model(inputs=res_model.input, outputs=res_model.get_layer('avg_pool').output)

        for phase in self.phases:
            vid_dir = self.base_dir + '/' + phase
            for i, v in enumerate(os.listdir(vid_dir)):
                feature_file = self.feature_dir + '/' + v.split('.')[0] + \
                               '_resnet.npy';
                if os.path.exists(feature_file):
                    continue
                video = imageio.get_reader(vid_dir + '/' + str(v), 'ffmpeg')
                print(vid_dir + '/' + str(v), i)
                vid_descriptors = np.zeros((999 * batch_size, 2048))
                frame_count = 0
                frame_ind = 0
                stop = False
                for b in range(999):
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
                    batch[:, :, :, 0] -= 103.939
                    batch[:, :, :, 1] -= 116.779
                    batch[:, :, :, 2] -= 123.68
                    vid_descriptors[b * batch_size:(b + 1) * batch_size] = model.predict_on_batch(batch).reshape(
                        batch.shape[0], -1)
                    if stop:
                        video.close()
                        break
                np.save(feature_file, vid_descriptors[:frame_count])
