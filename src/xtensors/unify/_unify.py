from __future__ import annotations
from typing import List, Literal, Optional, Sequence, Tuple, cast
import numpy as np
from xarray import DataArray

from xtensors.typing import DimLike, DimsLike, AxisDimPair
from xtensors.typing import NDArray


# def dims_to_axes(dims: Tuple[str,...], target_dims: Tuple[str,...]) -> Tuple[int,...]:
#     '''
#     Given a tuple of strings, find the indices where target_dims occur. 
#     '''
#     res: List[int] = []
#    
#     _dims = np.array(dims)
#
#     for tdim in target_dims:
#         index = np.where(_dims == tdim)[0]
#
#         index = index.item()
#         res.append(index)
#
#     return tuple(res)


def dim_to_axis(dims: Tuple[str,...], target_dim: str) -> int:
    '''
    Given a tuple of strings, find the index where target_dim occurs. If there
    is not exactly one occurrence, an IndexError is raised.

    '''
    _dims = np.array(dims)
    
    index = np.where(_dims == target_dim)[0]

    if len(index) == 0:
        raise IndexError(f'dim {target_dim} is not found')

    if len(index) > 1:
        raise IndexError(f'dim {target_dim} is ambiguous')

    index = index.item()
    return index


def get_axis_from_dim(x: NDArray, dim: str, fallback_axis: Optional[int]=None) -> int:
    '''
    Given a single dimension (string) and an array x, try locating the
    corresponding axis (int). If not found, the fallback is returned if
    provided, else an IndexError is raised.
    '''
    try:
        if isinstance(x, DataArray):
            return dim_to_axis(cast(Tuple[str,...], x.dims), dim)

        raise IndexError(f'trying to locate dimension {dim} but array/tensor is not named')

    except IndexError as e:
        if fallback_axis is not None:
            return fallback_axis
        
        raise IndexError(*e.args)

def get_axes(x: NDArray, dim: DimLike|DimsLike|None) -> Tuple[int,...]:
    '''
    Given dimensions (in strings) or axes (in integers), return a list of
    integers corresponding to the dimensions/axes

    Parameters:
        x: Input array
        dim: dimensions specifications, e.g.:
            2
            'N'
            (2, 'N') - 'N', or 2 if 'N' is not found
            (2, 3, 4)
            ('N', 'C', 'H')
            ((2, 'N'), (3, 'C'), (4, 'H'))

    '''

    if dim is None:
        return tuple([i for i in range(len(x.shape))])

    if isinstance(dim, int):
        return (dim,)

    if isinstance(dim, str):
        return (get_axis_from_dim(x, dim),)
    
    if isinstance(dim, tuple):
        if isinstance(dim[0], int):
            if isinstance(dim[1], str):
                return (get_axis_from_dim(x, dim[1], fallback_axis=dim[0]),)
            
            return cast(Tuple[int,...], dim)

        if isinstance(dim[0], str):
            dim = cast(Tuple[str,...], dim)
            return tuple(get_axis_from_dim(x, _dim) for _dim in dim)

        if isinstance(dim[0], tuple):
            dim = cast(Tuple[AxisDimPair,...], dim)
            return tuple(get_axis_from_dim(x, _dim[1], fallback_axis=_dim[0]) for _dim in dim)

    raise ValueError('Invalid argument combination')


def strip_dims(olddims: Tuple[str, ...], axis: Tuple[int, ...]) -> Tuple[str,...]:
    '''
    (For named dimensions) Remove dimensions corresponding to [axis] from named
    dimensions.
    '''
    newdims = list(olddims)
    stripped_dims = [olddims[i] for i in axis] 

    for sd in stripped_dims: newdims.remove(sd)

    return tuple(newdims)


def to_dataarray(x: NDArray) -> DataArray:
    if isinstance(x, DataArray): return x
    return DataArray(x)


def rename(x: NDArray, dim: DimLike, newname: str) -> DataArray:
    '''
    Rename a dimension. If the original array is unnamed, a named array will be
    returned.
    '''
    if isinstance(x, DataArray):
        _axis = get_axes(x, dim)[0]
        _dim = x.dims[_axis]
        return x.rename({_dim: newname})

    _x = x.__array__()
    _x = DataArray(_x)

    if isinstance(dim, int):
        return _x.rename({_x.dims[dim]: newname})

    if isinstance(dim, tuple):
        return _x.rename({_x.dims[dim[0]]: newname})

    raise ValueError


def new_axes(x: NDArray, *dims: str, position: Literal['before','after']='after') -> DataArray:
    _x = to_dataarray(x)
    if position == 'after':
        for dim in dims:
            _x = new_axis(_x, dim, position='after')
        return _x

    if position == 'before':
        for dim in dims[::-1]:
            _x = new_axis(_x, dim, position='before')
        return _x

    raise ValueError(f'Invalid position: {position}')


def new_axis(x: NDArray, dim: str, position: Literal['before','after']='after') -> DataArray:
    _x = to_dataarray(x)
    dims = list(_x.dims)
    _x = x.__array__()

    if position == 'before':
        _x = _x[None,...]
        _x = DataArray(_x, dims=['dims_appended_tmp'] + dims)
        _x = rename(_x, 0, dim)
        return _x

    if position == 'after':
        _x = _x[...,None]
        _x = DataArray(_x, dims=dims + ['dims_appended_tmp'])
        _x = rename(_x, -1, dim)
        return _x

    raise ValueError(f'Invalid position: {position}')
