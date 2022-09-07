from __future__ import annotations
'''
Functionalities with np arrays
'''
from typing import List, Sequence, Union
import numpy as np
from numpy.typing import NDArray


AxesLike = Union[int,Sequence[int],None]


def _axes_list(a: np.ndarray, axes: AxesLike) -> List[int]:
    if axes is None: return [i for i in range(len(a.shape))]

    if isinstance(axes, int): return [axes]

    return list(axes)


def axes_last(a: np.ndarray, axes: AxesLike=None) -> np.ndarray:

    axes = _axes_list(a, axes)
    newaxes = [i for i in range(len(a.shape)) if i not in axes] + axes

    return a.transpose(*newaxes)


def flatten(a: np.ndarray, axes: AxesLike=None) -> np.ndarray:
    b = axes_last(a, axes)
    axes = _axes_list(a, axes)
    n_preserved_axes = len(b.shape) - len(axes)
    
    return b.reshape(*b.shape[:n_preserved_axes], -1)


def fold_indices(indices: NDArray[np.int_], fold_shape: Sequence[int]) -> NDArray[np.int_]:
    '''
        indices: (*)
        fold_shape: [m1, m2, m3, ..., mN]

        return: folded indices (*, N)
    '''
    _indices = []
    indices = indices.copy()

    for l in fold_shape[::-1]:
        _indices.append(indices % l)
        indices //= l

    return np.array(_indices[::-1]).transpose(*[i+1 for i in range(len(indices.shape))], 0)
        
