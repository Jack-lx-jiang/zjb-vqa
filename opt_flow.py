import multiprocessing
import os
import sys
import threading

import click
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from skimage import transform
from tqdm import tqdm

from util.optical_flow import batch_opt_flow


@click.group()
def cli():
    pass


@cli.command()
@click.option('--video_dir', prompt=True)
@click.option('--output_dir', prompt=True)
@click.option('--threads', default=multiprocessing.cpu_count())
@click.option('--every', default=1)
@click.option('--limit', default=0)
def generate_opt_flow(video_dir, output_dir, threads, every, limit):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    listdir = os.listdir(video_dir)
    chunks = [listdir[i::threads] for i in range(threads)]

    tot_bar = tqdm(total=len(listdir))

    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=batch_opt_flow,
                             args=(chunks[i], video_dir, output_dir, every, limit, i + 1, tot_bar))
        t.start()
        thread_list.append(t)

    for thread in thread_list:
        thread.join()


@cli.command()
@click.option('--opt_dir', prompt=True)
@click.option('--output_dir', prompt=True)
@click.option('--feature', prompt=True)
# this extract function has some flaws....
# cannot use preprocess_input from keras because it uses imagenet's mean statistics
def generate_opt_feature(opt_dir, output_dir, feature):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    opts = os.listdir(opt_dir)
    res_model = ResNet50(weights='imagenet')
    pre_model = Model(inputs=res_model.input, outputs=[res_model.get_layer(feature).output])

    for i, o in enumerate(opts):
        ff = output_dir + '/' + o.split('.')[0] + '_' + feature + '_resnet.npy'
        if os.path.exists(ff):
            continue
        print(opt_dir + '/' + str(o), i)

        opt = np.load(opt_dir + '/' + o)
        batch = [transform.resize(opt['arr_0'][i], (224, 224), preserve_range=True) for i in
                 range(opt['arr_0'].shape[0])]
        batch = np.stack(batch, 0)
        batch = preprocess_input(batch)
        vid_descriptor = pre_model.predict_on_batch(batch)
        try:
            np.save(ff, vid_descriptor)
        except Exception as e:
            print(e)
            if os.path.exists(ff):
                os.remove(ff)
            sys.exit()


if __name__ == '__main__':
    cli()
