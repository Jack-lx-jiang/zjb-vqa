import os
import numpy as np
from tqdm import tqdm


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def make_embedding_weight(tokenizer):
    GLOVE_DIR = '/Users/KaitoHH/Downloads'
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))
    with tqdm(total=1917494) as pbar:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            pbar.update(1)
    embedding_matrix = np.zeros((len(tokenizer.index_word) + 1, 300))
    for idx, word in tokenizer.index_word.items():
        if word.endswith("'s"):
            word = word[:-2]
        if embeddings_index.get(word) is not None:
            embedding_matrix[idx] = embeddings_index[word]
        else:
            print(word)
    np.save('embedding.npy', embedding_matrix)
