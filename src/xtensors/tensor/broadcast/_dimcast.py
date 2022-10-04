from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Tuple, List
    from .._base import XTensor
    from ..typing import AxesPermutation
    from ._types import Dimcaster


def unilateral_dimcast(strict: bool=False) -> Dimcaster:
    def _dimcast(X: XTensor, Y: XTensor) -> Tuple[AxesPermutation, AxesPermutation]:
        axesp = castdim(X, Y, strict=strict)
        return [i for i in range(X.rank)], axesp
    return _dimcast

def trivial_dimcast(X: XTensor, Y: XTensor):
    axes_x: List[int|None] = [i for i in range(X.rank)]
    axes_y: List[int|None] = [i for i in range(Y.rank)]
    return axes_x, axes_y


def castdim(target: XTensor, subject: XTensor, strict: bool=False) -> AxesPermutation:
    '''
    Cast the dimensions of subject so that the two tensors are ready to be
    broadcast together.
    '''

    dims_t = list(target.dims)
    dims_s = list(subject.dims)
    max_unmatched_named_dimension_migration_index = len(subject.dims) - len(target.dims)

    # pad unnamed singletons so that the target tensor does not have fewer
    # dimensions than the subject tensor
    if len(dims_t) < len(dims_s):
        if strict:
            raise ValueError(f'Dimcast impossible with dimensions {dims_t} {dims_s} with strict=True')
        dims_t = [None for _ in range(len(dims_t), len(dims_s))] + dims_t
    # else:
    #     dims_s = [None for _ in range(len(dims_s), len(dims_t))] + dims_s

    ts_map = dict()

    # match named dimensions
    for axis_t, dim_t in enumerate(dims_t):
        if dim_t is not None:
            if dim_t in dims_s:
                ts_map[axis_t] = subject.get_axis(dim_t)

    axes_s_not_mached = [axis for axis in range(len(dims_s)) if axis not in ts_map.values()]


    dimcast_possible = True

    axes = []
    try:
        # match remaning
        for axis_t, dim_t in enumerate(dims_t):
            if axis_t not in ts_map.keys():
                # axis in target tensor is not yet matched
                axis_s = axis_t - len(dims_t) + len(dims_s)
                if axis_s >= 0:
                    dim_s = dims_s[axis_s]
                    if axis_s not in ts_map.values():
                        # axis in subject tensor is not yet matched
                        if dim_t is None or dim_s is None:
                            ts_map[axis_t] = axis_s
                        else:
                            raise ValueError()
                    else:
                        # axis in subject tensor is already matched
                        if axis_t < max_unmatched_named_dimension_migration_index:
                            axis_s = axes_s_not_mached.pop(0)
                            if dims_s[axis_s] is not None:
                                ts_map[axis_t] = axis_s
                            else:
                                raise ValueError()
                        else:
                            ts_map[axis_t] = None
                else:
                    ts_map[axis_t] = None
                    
        axes = [ts_map[axis_t] for axis_t in range(len(dims_t))]

        for axis in range(len(subject.shape)):
            if axis not in axes: raise ValueError()

    except ValueError:
        dimcast_possible = False 
    
    if not dimcast_possible:
        raise ValueError(f'Dimcast impossible with dimensions {dims_t} {dims_s}')

    return axes



