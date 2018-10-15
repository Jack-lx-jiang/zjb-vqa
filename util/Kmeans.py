import os
import random

import numpy as np
from sklearn.cluster import KMeans


def calculate_cluster_centers(feature_dir, feature, nb_centers, nb_feat, nb_jobs=8):
    feates = os.listdir(feature_dir)
    feat_idxes = []
    for i, f in enumerate(feates):
        if feature in f:
            feat_idxes.append(i)
    random.shuffle(feat_idxes)
    feat_selected = feat_idxes[:nb_feat]

    feat_sum = []
    for i in feat_selected:
        cur_feat = np.load(feature_dir + '/' + feates[i])
        cur_feat = cur_feat.reshape((-1, cur_feat.shape[-1]))
        feat_sum.append(cur_feat)
    feat_sum = np.concatenate(feat_sum)

    print('start kmeans')
    kmeans = KMeans(nb_centers, n_jobs=nb_jobs)
    kmeans.fit(feat_sum)
    print('kmeans finishes')

    path = feature_dir + '/kmeans.npy'
    np.save(path, kmeans.cluster_centers_)

# calculate_cluster_centers('dataset2/feature_avg_pool_activation_40_maxpool2_len100_inter15', 'activation_40_maxpool2', 128, 100)
