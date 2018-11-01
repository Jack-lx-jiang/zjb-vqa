import os

import numpy as np
from sklearn.cluster import KMeans


def calculate_cluster_centers(feature_dir, feature, nb_centers, nb_feat, output_dir=None, nb_jobs=8):
    feates = os.listdir(feature_dir)
    # feat_idxes = []
    ffs = set()
    for f in feates:
        f_dir = f.split('_')[0] + '_' + feature + '_resnet.npy'
        if os.path.exists(feature_dir + '/' + f_dir):
            ffs.add(f_dir)
            if len(ffs) == nb_feat:
                break

    feat_sum = []
    for f in ffs:
        cur_feat = np.load(feature_dir + '/' + f)
        cur_feat = cur_feat.reshape((-1, cur_feat.shape[-1]))
        feat_sum.append(cur_feat)
    feat_sum = np.concatenate(feat_sum)
    print(feat_sum.shape)

    print('start kmeans')
    kmeans = KMeans(nb_centers, n_jobs=nb_jobs)
    kmeans.fit(feat_sum)
    print('kmeans finishes')

    if output_dir != None:
        path = output_dir
    else:
        path = feature_dir + '/kmeans_' + str(nb_centers) + '.npy'
    np.save(path, kmeans.cluster_centers_)

