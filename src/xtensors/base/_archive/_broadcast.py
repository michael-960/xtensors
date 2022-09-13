from __future__ import annotations
import numpy as np
from numpy import ndarray
from typing import Protocol, Tuple, cast

from torch import wait
from xtensors.typing import NDArray
from xarray import DataArray


def are_shapes_broadcastable(shape1: Tuple[int,...], shape2: Tuple[int,...]) -> bool:
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def broadcast_shapes(shape1: Tuple[int,...], shape2: Tuple[int,...]) -> Tuple[int,...]:
    if len(shape1) < len(shape2): return broadcast_shapes(shape2, shape1)

    if not are_shapes_broadcastable: raise IndexError(f'shapes {shape1} and {shape2} cannot be broadcast')

    shape = list(shape1)[::-1]

    for axis, s2 in enumerate(list(shape2)[::-1]):
        if s2 != 1: shape[axis] = s2

    return tuple(shape[::-1])


def are_arrays_broadcastable(x: NDArray, y: NDArray) -> bool:
    if isinstance(x, DataArray) and isinstance(y, DataArray): return are_xarrays_broadcastable(x, y)

    return are_shapes_broadcastable(x.shape, y.shape)


def are_xarrays_broadcastable(x: DataArray, y: DataArray) -> bool:
    '''
    The broadcastability of two xarray.DataArrays is limited to only special
    cases for stability reasons. Two DataArrays x and y are broadcastable if
    and only if:

        x.dims contain y.dims, and the corresponding axes are compatible

        or

        y .dims contain x.dims, and the corresponding axes are compatible


    '''
    dim_x = x.dims
    dim_y = y.dims
    
    if len(dim_x) >= len(dim_y):
        if all([dim in dim_x for dim in dim_y]):
            for dim in dim_y:
                axis_x = cast(Tuple[int], x.get_axis_num((dim,)))[0]
                axis_y = cast(Tuple[int], y.get_axis_num((dim,)))[0]
                if x.shape[axis_x] != y.shape[axis_y] and x.shape[axis_x] != 1 and y.shape[axis_y] != 1:
                    return False
            return True
        return False

    return are_xarrays_broadcastable(y, x)


def broadcast_xarrays(x: DataArray, y: DataArray) -> Tuple[ndarray, ndarray, Tuple[str,...], Tuple[int,...]]:
    '''
    Determine how x and y should be broadcast. 

    Broadcastability is first checked by are_xarrays_broadcastable(x, y), if
    fails, an IndexError is raised.

    If x has more or equal dimensions than y, then the output dims will be
    conformed to x's, and vice versa.

    Return:
        A tuple of four objects:

            - ndarray of x
            - ndarray of y 
            (ready to be broadcast together)

            - output dimension names
            - output shape, e.g.

        x: dim (X, Y, Z, W) shape (1, 8, 32, 10)
        y: dim (X, W, Z) shape (9, 10, 1)

        => ndarray(1,8,32,10), ndarray(9,1,1,10), (X, Y, Z, W), (9, 8, 32, 10)
    '''
    if not are_xarrays_broadcastable(x, y): raise IndexError('xarrays cannot be broadcast')

    if len(x.dims) < len(y.dims):
        return broadcast_xarrays(y, x)

    
    _dims = cast(Tuple[str,...], x.dims)

    _x = x.__array__()
    _y = y.__array__() # has less axes than _x

    _shape = []

    _y_ax_map = dict()

    for dim in _dims:
        # axis_x will be the [axis] globally
        axis_x = cast(Tuple[int], x.get_axis_num((dim,)))[0]

        if dim in y.dims:
            axis_y = cast(Tuple[int], y.get_axis_num((dim,)))[0]
            _shape.append(max(x.shape[axis_x], y.shape[axis_y]))

            _y_ax_map[axis_x] = axis_y # map axis_y -> axis_x for axis permutation

        else: # there is no corresponding axis in y
            _y = _y[..., None] # add one axis
            _shape.append(x.shape[axis_x])

            _y_ax_map[axis_x] = len(_y.shape) - 1 # map (shape=1 new axis) -> axis_x

    _y_transpose_axes = tuple(_y_ax_map[_ax] for _ax in range(len(_y.shape)))
    _y = _y.transpose(_y_transpose_axes)

    _shape = tuple(_shape)
    return _x, _y, _dims, _shape



def broadcast_arrays(x: NDArray, y: NDArray) -> Tuple[ndarray, ndarray, Tuple[str,...]|None, Tuple[int,...]]:
    '''
    Try to broadcast two arrays.

    Case 1: x and y are both named (i.e. xarrays):

        returns broadcast_xarrays(x, y)

    Case 2: x and y are both unnamed (i.e. torch tensors or np ndarrays):

        use the default broadcasting logic for numpy arrays

    Case 3: the named tensor has more axes:
        
        the returned dimension names and shape are copied from the named tensor

    Caes 4: the unnamed tensor has more axes:

        e.g. x (64, 32, 32, 8), y (A=32, B=32, C=8)

        the returned dimension names and shape will be (dim0=64, A=32, B=32, C=8)
            
    '''
    if not are_arrays_broadcastable(x, y):
        raise IndexError('arrays cannot be broadcast')

    if isinstance(x, DataArray) and isinstance(y, DataArray):
        return broadcast_xarrays(x, y)



    _x = x.__array__()
    _y = y.__array__()


    shape = broadcast_shapes(x.shape, y.shape)

    if (not isinstance(x, DataArray)) and (not isinstance(y, DataArray)):
        return _x, _y, None, shape

    if isinstance(x, DataArray):
        _named = x 
    else:
        _named = y

    _named = cast(DataArray, _named)
    named_dims = cast(Tuple[str,...], _named.dims)

    if len(x.shape) > len(y.shape):
        _longer = x
    else:
        _longer = y

    if _named is _longer:
        dims = named_dims
    else:
        # named tensor is shorter
        dims = [f'dim_{i}' for i in range(len(_longer.shape))]
        dims[-len(named_dims):] = named_dims
        dims = tuple(dims)

    return _x, _y, dims, shape


    
