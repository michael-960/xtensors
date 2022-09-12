from typing import Literal, Optional, Sequence, Tuple, TypeVar, cast

from xarray import DataArray
from ..typing import NDArray, DimLike, DimsLike, AxisDimPair
import numpy as np

from ._base import is_named

from ._conv import to_dataarray, to_ndarray



def get_dims(x: NDArray) -> Tuple[str|None,...]:
    '''
    Return a list of dimnesion names for the given tensor
    Unnamed dimensions yield None
    '''

    if isinstance(x, DataArray):
        _dims = []
        for dim in x.dims:
            dim = cast(str, dim)

            if not is_named(dim):
                _dims.append(None)
            else:
                _dims.append(dim)

        return tuple(_dims)

    return tuple(None for _ in x.shape)


T = TypeVar('T')
def find_index(objs: Sequence[T], target: T) -> int:
    '''
    Given a tuple of objects, find the index where target occurs. If there
    is not exactly one occurrence, an IndexError is raised.
    '''
    _dims = np.array(objs)
    
    index = np.where(_dims == target)[0]

    if len(index) == 0:
        raise IndexError(f'{target} is not found')

    if len(index) > 1:
        raise IndexError(f'{target} is ambiguous')

    index = index.item()
    return index


def get_axis_from_dim(x: NDArray, dim: str, fallback_axis: Optional[int]=None) -> int:
    '''
    Given a single dimension (string) and an array x, try locating the
    corresponding axis (int). If not found, the fallback is returned if
    provided, else an IndexError is raised.
    '''


    try:
        if not is_named(dim): raise IndexError(f'{dim} is unnamed')

        if isinstance(x, DataArray):
            return find_index(cast(Tuple[str,...], x.dims), dim)
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


def rename(x: NDArray, dim: DimLike, newname: str) -> DataArray:
    '''
    Rename a dimension. If the original array is unnamed, a named array will be
    returned.
    '''

    if not is_named(newname): raise ValueError(f'Invalid dimension name: {newname}')

    x_ = to_dataarray(x)
    _axis = get_axes(x_, dim)[0]
    _dim = get_dims(x)[_axis]
    return x_.rename({_dim: newname})


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
    _x = to_ndarray(_x)

    if position == 'before':
        _x = _x[None,...]
        _x = DataArray(_x, dims=['__dims_appended_tmp'] + dims)
        _x = rename(_x, 0, dim)
        return _x

    if position == 'after':
        _x = _x[...,None]
        _x = DataArray(_x, dims=dims + ['__dims_appended_tmp'])
        _x = rename(_x, -1, dim)
        return _x

    raise ValueError(f'Invalid position: {position}')
