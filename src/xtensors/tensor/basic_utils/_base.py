from __future__ import annotations
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .._base import XTensor
    from ..typing import Array


def to_xtensor(x: XTensor|Array):
    from .._base import XTensor
    return x if isinstance(x, XTensor) else XTensor(x.__array__())
