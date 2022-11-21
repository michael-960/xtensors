from ._base import to_xtensor

from ._axes import (
        permute, newdims, align, shapes_broadcastable,
        permutation_well_defined, shape, stack, rank, index)

from ._dims import (
        mergedims, flatten, dimsfirst, dimslast,
        name_dim_if_absent, dims)

from ._coords import mergecoords, coords_same

from ._generalize import generalize_at_0, generalize_at_1, generalize_at_2, generalize_at_3


from ._misc import strip, copy_sig

