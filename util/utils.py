import os
import pickle

import numpy as np
from tqdm import tqdm


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def save(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def make_embedding_weight(tokenizer):
    GLOVE_DIR = '/Users/KaitoHH/Downloads'
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'), encoding='utf-8')
    with tqdm(total=1917494) as pbar:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            pbar.update(1)
    related_embeddings_index = {}
    for idx, word in tokenizer.index_word.items():
        if word.endswith("'s"):
            word = word[:-2]
        if embeddings_index.get(word) is not None:
            related_embeddings_index[word] = embeddings_index[word]
        else:
            print(word)
    save('embedding_index.pkl', related_embeddings_index)


def load_embedding_weight(tokenizer):
    embeddings_index = load('embedding_index.pkl')
    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
    for word, idx in tokenizer.word_index.items():
        if word.endswith("'s"):
            word = word[:-2]
        if embeddings_index.get(word) is not None:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            print(word)
    return embedding_matrix


def calculate_mean_std(feature_dir, feature):
    feates = os.listdir(feature_dir)
    count = 0
    feature_sum = None
    for f in feates:
        if feature in f:
            cur_feat = np.load(feature_dir + '/' + f)
            feature_sum = np.zeros(cur_feat.shape[1:])
            break

    for f in feates:
        if feature in f:
            cur_feat = np.load(feature_dir + '/' + f)
            feature_sum += cur_feat.sum(0)
            count += cur_feat.shape[0]
    mean = feature_sum / count

    print(count)
    print(feature_sum.shape)
    var = np.zeros_like(mean)
    mean_reshape = mean.reshape((1,) + mean.shape)
    for f in feates:
        if feature in f:
            cur_feat = np.load(feature_dir + '/' + f)
            var += np.sum((cur_feat - mean_reshape) ** 2, axis=0)

    var = var / count

    std = var ** 0.5

    np.save(feature_dir + '/' + feature + '_mean.npy', mean)
    np.save(feature_dir + '/' + feature + '_std.npy', std)
    return mean, std
