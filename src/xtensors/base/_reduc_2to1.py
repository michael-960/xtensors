from __future__ import annotations

import numpy as np

from .. import tensor as xtt


@xtt.generalize_at_0
def diagonal(X: xtt.XTensor, /, dim1: xtt.DimLike, dim2: xtt.DimLike, dim_out: str|None) -> xtt.XTensor:
    '''
        Reduce a tensor x by taking the diagonal elements along [dim1] and
        [dim2]
    '''

    axis1 = X.get_axis(dim1)
    axis2 = X.get_axis(dim2)

    _y = np.diagonal(X.data, axis1=axis1, axis2=axis2)

    return xtt.XTensor(_y, 
            dims=xtt.strip(X.dims, [axis1, axis2]) + [dim_out],
            coords=xtt.strip(X.coords, [axis1, axis2]) + [None],
            )

