import datetime
from subprocess import call

from dataset import Dataset

# This needs to be changed
data_dir = '../data/DatasetA'


def main():
    # generate feature (this should run once)
    print('Generating video feature...')
    d = Dataset(base_dir=data_dir)
    d.compute_frame_feature()

    exp_name = 'experiments/encode_decode_model/'

    # train model
    print('Now training model...')
    call(
        ['python', 'vqa.py', '--data_dir', data_dir, '--exp_name', exp_name, 'encode_decode_model', 'train', '--epoch',
         '20'])

    # test model
    print('Now testing model...')
    call(['python', 'vqa.py', '--data_dir', data_dir, '--exp_name', exp_name, 'encode_decode_model', 'test'])

    # copy submit.txt
    call(['cp', 'submit.txt', "../submit/submit_" + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"])
    print('Finished.')


if __name__ == '__main__':
    main()
