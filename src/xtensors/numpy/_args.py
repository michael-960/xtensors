import numpy as np
from numpy.typing import NDArray

from ._np import AxesLike, flatten, fold_indices, _axes_list



def argsmax(a: np.ndarray, axes: AxesLike=None) -> NDArray[np.int_]:
    b = flatten(a, axes)
    args = np.argmax(b, axis=-1)
    return fold_indices(args, [a.shape[i] for i in _axes_list(a, axes)])


def argsmin(a: np.ndarray, axes: AxesLike=None) -> NDArray[np.int_]:

    b = flatten(a, axes)
    args = np.argmin(b, axis=-1)

    return fold_indices(args, [a.shape[i] for i in _axes_list(a, axes)])


def nanargsmax(a: np.ndarray, axes: AxesLike=None) -> NDArray[np.int_]:

    b = flatten(a, axes)
    args = np.nanargmax(b, axis=-1)

    return fold_indices(args, [a.shape[i] for i in _axes_list(a, axes)])


def nanargsmin(a: np.ndarray, axes: AxesLike=None) -> NDArray[np.int_]:

    b = flatten(a, axes)
    args = np.nanargmin(b, axis=-1)

    return fold_indices(args, [a.shape[i] for i in _axes_list(a, axes)])
