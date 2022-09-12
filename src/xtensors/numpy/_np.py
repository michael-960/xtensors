from __future__ import annotations
'''
Functionalities with np arrays
'''
from typing import List, Literal, Sequence, Union, cast
import numpy as np
from numpy.typing import NDArray


AxesLike = Union[int,Sequence[int],None]


def _axes_list(a: np.ndarray, axes: AxesLike) -> List[int]:
    if axes is None: return [i for i in range(len(a.shape))]

    if isinstance(axes, int): return [axes]

    return list(axes)


def axes_last(a: np.ndarray, axes: AxesLike=None) -> np.ndarray:
    '''
    Reshape the array so that the specified axes are placed at the end
    '''
    axes = _axes_list(a, axes)
    newaxes = [i for i in range(len(a.shape)) if i not in axes] + axes

    return a.transpose(*newaxes)


def axes_first(a: np.ndarray, axes: AxesLike=None) -> np.ndarray:
    '''
    Reshape the array so that the specified axes are placed at the beginning (left)
    '''
    axes = _axes_list(a, axes)
    newaxes = axes + [i for i in range(len(a.shape)) if i not in axes]

    return a.transpose(*newaxes)


def flatten(a: np.ndarray, axes: AxesLike=None, position: Literal['left', 'right']='right') -> np.ndarray:
    axes = _axes_list(a, axes)
    n_preserved_axes = len(a.shape) - len(axes)
    

    if position == 'right':
        b = axes_last(a, axes)
        return b.reshape(*b.shape[:n_preserved_axes], -1)
    else:
        b = axes_first(a, axes)
        return b.reshape(-1, *b.shape[-n_preserved_axes:])


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
        


def bincount(x: NDArray[np.int_], N: int|None=None) -> NDArray[np.int_]:
    '''
        x: (*, M)
        return: (*, N)
    '''

    if N is None:
        N = np.max(x) + 1

    shape = x.shape[:-1]

    # (D, M)
    _x = flatten(x, [a for a in range(len(x.shape))][:-1], position='left')

    D = _x.shape[0]
    index = np.arange(D).reshape(D, 1)


    N = cast(int, N)

    y = np.bincount((index*N + _x).reshape(-1), minlength=N*D).reshape(*shape, N)

    return y



    
    
