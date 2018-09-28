import os
import pickle as p
import re
import sys
import time
from collections import Counter

import imageio
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint
from keras.models import Model
from skimage import transform


class BaseModel():
    def __init__(self):
        reg = re.compile(r"(?<=inter)\d+")
        # print(self.feature_dir)
        dataset_interval = int(reg.search(self.feature_dir).group(0))
        assert dataset_interval <= self.interval
        if not os.path.exists(self.feature_dir):
            print('Features not found')
            print('Generating feature.....')
            self.preprocess_video()
        self.dataset.set_config(self.data_dir, self.minimum_appear, self.max_video_len, self.max_question_len,
                                self.train_threshold, self.interval // dataset_interval, self.feature_dir,
                                self.features)
        self.exp_name = ''

    def train(self, batch_size, epoch):
        model = self.build()
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d_%H-%M-%S", time_local)
        self.exp_name = 'experiments/{}/{}/'.format(self.model_name, dt)
        ename = self.exp_name
        data = self.dataset
        if not os.path.exists(ename):
            os.makedirs(ename)
        trained = model.fit_generator(data.generator('train', batch_size), data.get_nb_steps('train', batch_size),
                                      epoch,
                                      validation_data=data.generator('val', batch_size),
                                      validation_steps=data.get_nb_steps('val', batch_size),
                                      # validation_data = dum_val,
                                      callbacks=[  # EarlyStopping(patience=5),
                                          ModelCheckpoint(
                                              ename + 'E{epoch:02d}-L{val_loss:.2f}-A{val_multians_accuracy:.2f}.pkl',
                                              save_weights_only=True,
                                              monitor='val_multians_accuracy',
                                              save_best_only=False,
                                              period=5)])
        p.dump(trained.history, open(ename + 'history.pkl', 'wb'))
        model.save_weights(ename + 'latest.pkl')
        # p.dump(self, open(ename + 'ModelInstance.pkl', 'wb'))

    def test(self, batch_size, exp_name, pkl_name='latest.pkl'):
        model = self.build()
        model.load_weights(exp_name + pkl_name)
        data = self.dataset
        vid, questions, _ = data.preprocess_text('test')
        prediction = model.predict_generator(data.generator('test', batch_size),
                                             steps=data.get_nb_steps('test', batch_size),
                                             verbose=1)
        prediction = np.argmax(prediction, axis=1)
        # get statistics of counter
        print(Counter(prediction))
        with open(exp_name + pkl_name.split('.')[0] + '_submit.txt', 'w') as f:
            for idx, v_id in enumerate(vid):
                s = [v_id]
                for jdx, question in enumerate(questions[idx * 5:idx * 5 + 5]):
                    answer = self.dataset.dict.idx2ans[prediction[idx * 5 + jdx]]
                    s.append(',{},{}'.format(question, answer))
                f.write(''.join(s) + '\n')

    # If you need to override the function, make sure the outputs' order is same as self.features
    def preprocess_video_model(self):
        res_model = ResNet50(weights='imagenet')
        pre_model = Model(inputs=res_model.input, outputs=[res_model.get_layer(f).output for f in self.features])
        return pre_model

    def preprocess_video(self):
        pre_model = self.preprocess_video_model()
        for phase in ['train', 'test']:
            vid_dir = self.data_dir + '/' + phase
            for i, v in enumerate(os.listdir(vid_dir)):
                feature_files = [self.feature_dir + '/' + v.split('.')[0] + '_' + f + '_resnet.npy' for f in
                                 self.features]
                if all([os.path.exists(ff) for ff in feature_files]):
                    continue
                print(vid_dir + '/' + str(v), i)
                video = imageio.get_reader(vid_dir + '/' + str(v), 'ffmpeg')
                # vid_descriptors = [np.zeros((self.max_video_len,)+o.shape[1:]) for o in pre_model.output]
                frame_count = 0
                frame_ind = 0
                batch = np.zeros((self.max_video_len, 224, 224, 3))
                for t in range(self.max_video_len):
                    try:
                        frame = video.get_data(frame_ind)
                    except (imageio.core.CannotReadFrameError, IndexError):
                        break
                    else:
                        frame = transform.resize(frame, (224, 224))
                        batch[t] = frame
                        frame_count += 1

                    frame_ind += self.interval
                batch = preprocess_input(batch)
                vid_descriptors = pre_model.predict_on_batch(batch)
                video.close()
                if not os.path.exists(self.feature_dir):
                    os.mkdir(self.feature_dir)
                try:
                    ffi = 0
                    for ff, vid_des in zip(feature_files, vid_descriptors):
                        np.save(ff, vid_des[ffi][:frame_count])
                        ffi += 1
                except Exception as e:
                    print(e)
                    gen = (ff for ff in feature_files if os.path.exists(ff))
                    for ff in gen:
                        os.remove(ff)
                    sys.exit()

    def generate_feature_dir_name(self):
        dir_name = self.data_dir + '/feature'
        for f in self.features:
            dir_name += '_' + f
        dir_name += '_len' + str(self.max_video_len)
        dir_name += '_inter' + str(self.interval)
        return dir_name

    def build(self):
        pass
