import io
import urllib, base64
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

import numbers
import array
from collections.abc import Iterable

import numpy as np
from scipy import linalg
import scipy.sparse as sp

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as util_shuffle
from sklearn.utils.random import sample_without_replacement

def createPlane( n_samples=100, n_features=100, *, n_informative=10, n_targets=1, bias=0.0,
        effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None, ):


    n_informative = min(n_features, n_informative)
    generator = check_random_state(random_state)

    if effective_rank is None:
        # Randomly generate a well conditioned input set
        X = generator.randn(n_samples, n_features)

    else:
        # Randomly generate a low rank, fat tail input set
        X = make_low_rank_matrix(
            n_samples=n_samples,
            n_features=n_features,
            effective_rank=effective_rank,
            tail_strength=tail_strength,
            random_state=generator,
        )

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = np.zeros((n_features, n_targets))
    ground_truth[:n_informative, :] = 100 * generator.rand(n_informative, n_targets)

    y = np.dot(X, ground_truth) + bias

    # Add noise
    if noise > 0.0:
        y += generator.normal(scale=noise, size=y.shape)

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]
        ground_truth = ground_truth[indices]

    y = np.squeeze(y)

    if coef:
        return X, y, np.squeeze(ground_truth)

    else:
        return X, y
