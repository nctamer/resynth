import os
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def refine_mu(magnitudes, median_multi=1.5):
    mu = magnitudes.sum(axis=0) / len(magnitudes)
    mu_dist = euclidean_distances(magnitudes, [mu]).reshape(-1)
    median = np.median(mu_dist)
    valid_bool = mu_dist < median_multi * np.median(mu_dist)
    print("mu:", mu)
    print("median", median)
    print("coverage:", valid_bool.sum() / valid_bool.__len__())
    return magnitudes[valid_bool]

multiplier = 1.8
hmag = np.load("/home/nazif/PycharmProjects/data.npy")
for i in range(20):
    hmag = refine_mu(hmag, median_multi=multiplier)
final_mu = hmag.sum(axis=0) / len(hmag)
#all_hmag = np.load("/home/nazif/PycharmProjects/data.npy")
#final_dist = euclidean_distances(all_hmag, [final_mu]).reshape(-1)
#final_valid = final_dist < multiplier * np.median(final_dist)
#final_valid.sum() / len(final_valid)


