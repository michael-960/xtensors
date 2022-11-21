from __future__ import annotations
from typing import TYPE_CHECKING, List, Literal, Sequence

from ... import numpy as xtnp

from ._axes import permute

from ._generalize import generalize_at_0

from ._misc import copy_sig


if TYPE_CHECKING:
    from .._base import XTensor, TensorLike
    from ..typing import DimsLike, DimLike, Dims


def mergedims(X: XTensor|Dims, Y: XTensor|Dims) -> Dims:
    '''
    '''

    from .._base import XTensor

    if isinstance(X, XTensor): dims_x = list(X.dims)
    else: dims_x = X

    if isinstance(Y, XTensor): dims_y = list(Y.dims)
    else: dims_y = Y


    rank_x, rank_y = len(dims_x), len(dims_y)

    if rank_x < rank_y: return mergedims(Y, X)

    newdims = []

    dims_y = [None for _ in range(rank_y, rank_x)] + dims_y
    
    for dim_x, dim_y in zip(dims_x, dims_y):
        if dim_x is not None and dim_y is not None and dim_x != dim_y:
            raise ValueError(f'Incompatible dimensions: {dims_x} and {dims_y}')
        newdims.append(dim_x if dim_x is not None else dim_y)

    return newdims


def dimslast(X: XTensor, dims: Sequence[str]) -> XTensor:
    """
        Move named dimensions to the right.

        :param X: target tensor
        :param dims: a sequence of dimension names

        :return: a new :py:class:`xtensors.XTensor` with :code:`dims` moved to the right.

    """
    axes: List[int] = [X.get_axis(dim) for dim in dims]
    other_axes: List[int|None] = [axis for axis in range(X.rank) if axis not in axes]

    return permute(X, other_axes+axes)


def dimsfirst(X: XTensor, dims: Sequence[str]) -> XTensor:
    """
        Move named dimensions to the left.

        :param X: target tensor
        :param dims: a sequence of dimension names

        :return: a new :py:class:`xtensors.XTensor` with :code:`dims` moved to the left

    """
    axes: List[int] = [X.get_axis(dim) for dim in dims]
    other_axes: List[int|None] = [axis for axis in range(X.rank) if axis not in axes]

    return permute(X, axes+other_axes)


def _flatten(X: TensorLike, /, dims: DimsLike, dim_out: str|None, *,
             position: Literal['left', 'right']='right') -> XTensor: ...

@copy_sig(_flatten)
@generalize_at_0
def flatten(X: XTensor, /,
        dims: DimsLike, dim_out: str|None, *,
        position: Literal['left', 'right']='right') -> XTensor:
    """
    Flatten a tensor

    :param X: target tensor
    :param dims: dimensions to be flattened
    :param dim_out: name of the new dimension
    :param position: :code:`left` or :code:`right`, where the new dimension is placed

    """

    from .._base import XTensor
    x = X.data
    # axes = [X.get_axis(dim) for dim in dims]
    axes = X.get_axes(dims)

    remaining_dims = [dim for axis, dim in enumerate(X.dims) if axis not in axes]
    remaining_coords = [coord for axis, coord in enumerate(X.coords) if axis not in axes]

    # TODO: implement coordinate meshgrid
    coord_out = None

    x_flat = xtnp.flatten(x, axes, position=position)

    if position == 'left':
        Y = XTensor(x_flat, [dim_out]+remaining_dims, [coord_out] + remaining_coords)
        return Y

    else:
        Y = XTensor(x_flat, remaining_dims+[dim_out], remaining_coords+[coord_out])
        return Y


def _name_dim_if_absent(
        X: TensorLike, /, axis: int, dim: str, *, force: bool=False
) -> XTensor: ...


@copy_sig(_name_dim_if_absent)
@generalize_at_0
def name_dim_if_absent(X: XTensor, /, axis: int, dim: str, *, force: bool=False) -> XTensor:
    """
    Ensure that :code:`X` has a named dimension called :code:`dim`. If not
    already, the dimension at :code:`axis` will be named so.

    :param X: target tensor
    :param axis: the axis to be renamed
    :param dim: axis name
    :param force: if :code:`True`, original axis name will be overwritten

    """
    X1 = X.viewcopy()
    if dim in X.dims:
        return X1

    if X1.dims[axis] is not None and not force:
        raise ValueError(f'Tensor already has its dimension named {X1.dims[axis]} at axis={axis}')

    X1.set_dim(axis, dim)
    return X1


@generalize_at_0
def dims(X: XTensor): return X.dims

