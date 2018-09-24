from dataset import Dataset
from subprocess import call
import datetime

# This needs to be changed
data_dir = '../data/DatasetA'


def main():
    # generate feature (this should run once)
    d = Dataset(base_dir=data_dir)
    d.compute_frame_feature()

    exp_name = 'experiments/encode_decode_model/'

    # train model
    call(['python3', 'vqa.py', '--data_dir', data_dir, '--exp_name', exp_name, 'encode_decode_model', 'train'])

    # test model
    call(['python3', 'vqa.py', '--data_dir', data_dir, '--exp_name', exp_name, 'test'])

    # copy submit.txt
    call(['cp', 'submit.txt', "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"])


if __name__ == '__main__':
    main()
