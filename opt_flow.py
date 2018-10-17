import multiprocessing
import os
import threading

import click
from tqdm import tqdm

from util.optical_flow import batch_opt_flow


@click.command()
@click.option('--video_dir', prompt=True)
@click.option('--output_dir', prompt=True)
@click.option('--threads', default=multiprocessing.cpu_count())
def generate_opt_flow(video_dir, output_dir, threads):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    listdir = os.listdir(video_dir)
    chunks = [listdir[i::threads] for i in range(threads)]

    tot_bar = tqdm(total=len(listdir))

    thread_list = []
    for i in range(threads):
        t = threading.Thread(target=batch_opt_flow, args=(chunks[i], video_dir, i + 1, tot_bar))
        t.start()
        thread_list.append(t)

    for thread in thread_list:
        thread.join()


if __name__ == '__main__':
    generate_opt_flow()
