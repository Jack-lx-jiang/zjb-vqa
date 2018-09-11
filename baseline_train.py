import pickle as p

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import Adadelta

from dataset import Dataset
from model.model import stacked_attention_model
from util.metrics import multians_accuracy

dataset = Dataset()
batch_size = 128

video = Input((dataset.max_video_len, dataset.frame_size))
question = Input((dataset.max_question_len,), dtype='int32')
# model = Model(inputs=[video, question],
#               outputs=base_model(video, question, dataset.vocabulary_size, dataset.max_question_len,
#                                  dataset.max_video_len, dataset.answer_size))
model = Model(inputs=[video, question],
              outputs=stacked_attention_model(video, question, dataset.vocabulary_size, dataset.max_question_len,
                                              dataset.answer_size))
model.compile(optimizer=Adadelta(), loss=binary_crossentropy, metrics=[multians_accuracy])
# exp_name = 'experiments/baseline_test'
exp_name = 'experiments/stacked_attention/test'

nb_step = 3325 * 5 // batch_size + 1
train_mod = True
if train_mod:
    trained = model.fit_generator(dataset.generator(batch_size, 'train'), nb_step, 100,
                                  validation_data=dataset.generator(batch_size, 'train'),
                                  validation_steps=nb_step / 10,
                                  callbacks=[EarlyStopping(patience=5),
                                             ModelCheckpoint(
                                                 exp_name + '_.{epoch:02d}-{val_loss:.2f}.pkl',
                                                 save_best_only=True)])
    p.dump(trained.history, open(exp_name + '_history.pkl', 'wb'))
    model.save_weights(exp_name + '.pkl')
else:
    model.load_weights('experiments/baseline_test_.77-0.00.pkl')
    metrics = model.evaluate_generator(dataset.generator(batch_size, 'train'), nb_step)
    # print(metrics)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))
