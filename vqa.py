import os, time
import pickle as p
from collections import Counter

import click
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta

from dataset import Dataset
from util.loss import focal_loss
from util.metrics import multians_accuracy

'''
Usage: vqa.py [OPTIONS] MODEL_NAME COMMAND [ARGS]...

where MODEL_NAME is defined in model/__init__.py
and COMMAND is one of [train, test, eval]

Example Usage:
vqa.py base_model train
vqa.py --data_dir path/to/data test
vqa.py --batch 256 train --epoch 200

Also See:
vqa.py --help
'''

dataset = None
batch_size = None
model = None


@click.group()
@click.argument('model_name')
@click.option('--exp_name', default=None, help='Path to store the model.')
@click.option('--data_dir', default=None, help='Data path containing extracted feature.')
@click.option('--batch', default=128, help='Size of one batch.')
@click.pass_context
def cli(ctx, model_name, exp_name, data_dir, batch):
    views = __import__('model')
    cur_model = getattr(views, model_name)

    if not exp_name:
        time_now = int(time.time())
        time_local = time.localtime(time_now)
        dt = time.strftime("%Y-%m-%d_%H:%M:%S", time_local)
        exp_name = 'experiments/{}/'.format(model_name+dt)
    ctx.obj = {'exp_name': exp_name}
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)

    global dataset, batch_size, model
    dataset = Dataset(base_dir=data_dir)
    batch_size = batch
    model = cur_model(dataset.vocabulary_size, dataset.max_question_len, dataset.max_video_len, dataset.frame_size,
                      dataset.answer_size)
    model.compile(optimizer=Adadelta(), loss=[focal_loss(alpha=.25, gamma=2)], metrics=[multians_accuracy])


@cli.command()
@click.option('--nb_step', default=0)
@click.option('--epoch', default=100)
@click.option('--interval', default=1)
@click.pass_context
def train(ctx, nb_step, epoch, interval):
    threds = 0.95
    if not nb_step:
        nb_step = 3325 * 5 * threds // batch_size + 1
    val_step = 3325 * 5 * (1 - threds) // batch_size
    log_dir = './logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    exp_name = ctx.obj['exp_name']
    # gen = (i for i in dataset.generator(batch_size, 'train'))
    # x, y = next(gen)
    # dum_val = (x, y)
    trained = model.fit_generator(dataset.generator(batch_size, 'train', interval, threds), nb_step, epoch,
                                  validation_data=dataset.generator(batch_size, 'val', interval, threds),
                                  validation_steps=val_step,
                                  # validation_data = dum_val,
                                  callbacks=[#EarlyStopping(patience=5),
                                             ModelCheckpoint(
                                                 exp_name + 'E{epoch:02d}-L{val_loss:.2f}.pkl',
                                                 save_best_only=True)])
    p.dump(trained.history, open(exp_name + 'history.pkl', 'wb'))
    model.save_weights(exp_name + 'latest.pkl')


@cli.command()
@click.pass_context
def test(ctx):
    exp_name = ctx.obj['exp_name']
    model.load_weights(exp_name + 'latest.pkl')
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


@cli.command()
@click.option('--nb_step', default=0)
@click.pass_context
def eval(ctx, nb_step):
    if not nb_step:
        nb_step = 3325 * 5 // batch_size + 1
    exp_name = ctx.obj['exp_name']
    model.load_weights(exp_name + 'latest.pkl')
    metrics = model.evaluate_generator(dataset.generator(batch_size, 'train'), nb_step)
    # print(metrics)
    for i in range(len(model.metrics_names)):
        print(str(model.metrics_names[i]) + ": " + str(metrics[i]))


if __name__ == '__main__':
    cli()
