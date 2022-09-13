from __future__ import annotations

from typing import Any
import numpy as np
import numpy.typing as npt

from .. import tensor as xtt
from scipy import special



@xtt.generalize_3
@xtt.promote_ternary_operator()
def where(condition: npt.NDArray, x: npt.NDArray, y: npt.NDArray, /) -> npt.NDArray:
    '''
    '''
    return np.where(condition, x, y)


@xtt.generalize_1
def softmax(X: xtt.XTensor, /, dim: xtt.DimLike) -> xtt.XTensor:
    axis = X.get_axis(dim)
    _y = special.softmax(X.data, axis=axis)
    return xtt.XTensor(_y, dims=X.dims, coords=X.coords)


def get_rank(x: Any) -> int:
    if hasattr(x, 'shape'):
        return len(x.shape)

    if hasattr(x, '__array__'):
        return get_rank(x.__array__())

    if isinstance(x, (list, tuple)):
        return get_rank(x[0]) + 1

    return 0

