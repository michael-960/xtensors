from __future__ import annotations

from typing import TYPE_CHECKING, List, Literal, Sequence, Tuple
from ..typing import AxesPermutation, Dims, Coords

if TYPE_CHECKING: from .._base import XTensor


def permutation_well_defined(axes: AxesPermutation, rank: int|None=None) -> bool:
    axes_nnon = [axis for axis in axes if axis is not None]
    condition = (len(axes_nnon) == len(set(axes_nnon)))

    if rank is not None:
        condition = condition and all([axis in axes_nnon for axis in range(rank)])

    return condition


def permute(X: XTensor, axes: AxesPermutation) -> XTensor:
    '''
    
    '''
    from .._base import XTensor

    data_ = X.data
    axes = axes.copy()
    newaxis = len(data_.shape)

    axes_refined: List[int] = []
    newdims = []
    newcoords = []
    
    for axis_t, axis_x in enumerate(axes):
        if axis_x is None:
            data_ = data_[...,None]
            axes_refined.append(newaxis)
            newdims.append(None)
            newcoords.append(None)
            newaxis += 1
        else:
            axes_refined.append(axis_x)
            newdims.append(X.dims[axis_x])
            newcoords.append(X.coords[axis_x])

    data_ = data_.transpose(*axes_refined)

    Y = XTensor(data_, dims=newdims, coords=newcoords)
    return Y


def newdims(X: XTensor,
        dims: Dims|None=None, coords: Coords|None=None,
        position: Literal['left','right']='left') -> XTensor:
    '''
    Pad singleton dimensions to the given tensor with the specified dimension
    names and coordinates.
    ''' 
    from .._base import XTensor
    if dims is not None and coords is not None:
        assert len(dims) == len(coords)

        
    _x = X.data

    if dims is not None:
        n_dims = len(dims)

    else:
        if coords is None: raise ValueError('dims and coords cannot both be None')
        n_dims = len(coords)

    if dims is None:
        newdims = [None for _ in range(n_dims)]
    else:
        newdims = dims

    if coords is None:
        newcoords = [None for _ in range(n_dims)]
    else:
        newcoords = coords


    if position == 'left':
        _y = _x.reshape(*[1 for _ in range(n_dims)], *_x.shape)
        return XTensor(_y, dims=newdims+list(X.dims), coords=newcoords+list(X.coords))

    else:
        _y = _x.reshape(*_x.shape, *[1 for _ in range(n_dims)])
        return XTensor(_y, dims=list(X.dims)+newdims, coords=list(X.coords)+newcoords)


def align(X: XTensor, Y: XTensor) -> Tuple[XTensor, XTensor]:
    '''
    Pad singleton unnamed dimensions to (at most) one of the two tensors so that the
    returned tensors have the same number of dimnesions.
    '''
    if len(X.shape) > len(Y.shape):
        return X, newdims(Y, [None for _ in range(len(Y.shape), len(X.shape))])
    
    if len(X.shape) < len(Y.shape):
        return newdims(X, [None for _ in range(len(X.shape), len(Y.shape))]), Y

    return X, Y


def shapes_broadcastable(a: Sequence[int], b: Sequence[int]) -> bool:
    for sa, sb in zip(a[::-1], b[::-1]):
        if sa != sb and sa != 1 and sb != 1: return False
    return True
