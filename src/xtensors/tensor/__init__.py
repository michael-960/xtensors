from ._base import XTensor

from .typing import BinaryOperator

from ._decors import with_broadcast, generalize_1, generalize_2, generalize_3


from .basic_utils import (
    permute, newdims, align, shapes_broadcastable,
    mergedims, flatten, dimsfirst, dimslast,
    mergecoords
    )
