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

def createCircle( n_samples=100, *, shuffle=True, noise=None, random_state=None, factor=0.8):
    

    if factor >= 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")

    if isinstance(n_samples, numbers.Integral):
        n_samples_out = n_samples // 2
        n_samples_in = n_samples - n_samples_out
    else:
        try:
            n_samples_out, n_samples_in = n_samples
        except ValueError as e:
            raise ValueError(
                "`n_samples` can be either an int or a two-element tuple."
            ) from e

    generator = check_random_state(random_state)
    # so as not to have the first point = last point, we set endpoint=False
    linspace_out = np.linspace(0, 2 * np.pi, n_samples_out, endpoint=False)
    linspace_in = np.linspace(0, 2 * np.pi, n_samples_in, endpoint=False)
    outer_circ_x = np.cos(linspace_out)                                                                                                          
    outer_circ_y = np.sin(linspace_out)                                                                                                          
    inner_circ_x = np.cos(linspace_in) * factor                                                                                                  
    inner_circ_y = np.sin(linspace_in) * factor                                                                                                  
                                                                                                                                                 
    X = np.vstack(                                                                                                                               
        [np.append(outer_circ_x, inner_circ_x), np.append(outer_circ_y, inner_circ_y)]                                                           
    ).T
    y = np.hstack(
        [np.zeros(n_samples_out, dtype=np.intp), np.ones(n_samples_in, dtype=np.intp)]
    )
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

    if noise is not None:
        X += generator.normal(scale=noise, size=X.shape)

    return X, y
