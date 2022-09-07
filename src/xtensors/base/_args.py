from __future__ import annotations

from xarray import DataArray
from ..typing import NDArray, DimLike, DimsLike

from ..unify import get_axes





def argmin(x: NDArray, dim: DimLike|DimsLike|None) -> DataArray:
    _axes = get_axes(x, dim)
