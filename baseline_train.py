import os
import pickle as p
from collections import Counter

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adadelta

from dataset import Dataset
from model.model import stacked_attention_model
from util.metrics import multians_accuracy

# collect env variable
dataset = Dataset()
batch_size = 128
nb_step = int(os.getenv('nb_step', 3325 * 5 // batch_size + 1))
epochs = int(os.getenv('epochs', 100))
# exp_name = 'experiments/baseline_test'
exp_name = 'experiments/stacked_attention/test'
if not os.path.exists(exp_name):
    os.makedirs(exp_name)

# create model
video = Input((dataset.max_video_len, dataset.frame_size))
question = Input((dataset.max_question_len,), dtype='int32')
# model = Model(inputs=[video, question],
#               outputs=base_model(video, question, dataset.vocabulary_size, dataset.max_question_len,
#                                  dataset.max_video_len, dataset.answer_size))
model = Model(inputs=[video, question],
              outputs=stacked_attention_model(video, question, dataset.vocabulary_size, dataset.max_question_len,
                                              dataset.answer_size))
model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])

train_mod = False
test_mod = True
if train_mod:
    trained = model.fit_generator(dataset.generator(batch_size, 'train'), nb_step, epochs,
                                  validation_data=dataset.generator(batch_size, 'train'),
                                  validation_steps=nb_step / 10,
                                  callbacks=[EarlyStopping(patience=5),
                                             ModelCheckpoint(
                                                 exp_name + '_.{epoch:02d}-{val_loss:.2f}.pkl',
                                                 save_best_only=True)])
    p.dump(trained.history, open(exp_name + '_history.pkl', 'wb'))
    model.save_weights(exp_name + '.pkl')
elif test_mod:
    model.load_weights(exp_name + '.pkl')
    vid, questions, _ = dataset.preprocess_text('test')
    total_steps = len(questions) // batch_size + 1
    prediction = model.predict_generator(dataset.generator(batch_size, 'test'), steps=total_steps, verbose=1)
    prediction = np.argmax(prediction, axis=1)
    # get statistics of counter
    print(Counter(prediction))
    with open('submit.txt', 'w') as f:
        for idx, v_id in enumerate(vid):
            s = [v_id]
            for jdx, question in enumerate(questions[idx * 5:idx * 5 + 5]):
                answer = dataset.dict.idx2ans[prediction[idx * 5 + jdx]]
                s.append(',{},{}'.format(question, answer))
            f.write(''.join(s) + '\n')
else:
    model.load_weights(exp_name + '.pkl')
    metrics = model.evaluate_generator(dataset.generator(batch_size, 'train'), nb_step)
    # print(metrics)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
