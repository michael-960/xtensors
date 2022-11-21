from __future__ import annotations
from collections.abc import Sequence
from numbers import Real
import numpy as np
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .._base import XTensor
    from ..typing import TensorLike


def to_xtensor(x: TensorLike) -> XTensor:
    """
    Coerce a :py:class:`xtensor.TensorLike` object to :py:class:`XTensor`

    """
    from .._base import XTensor
    if isinstance(x, XTensor): return x
    
    try:
        return XTensor(x.__array__()) # type: ignore
    except AttributeError: pass

    if isinstance(x, Real):
        return XTensor(np.array(x))

    if isinstance(x, Sequence):
        return XTensor(np.array(list(x)))

    raise TypeError(f'Object cannot be safely converted to XTensor')

