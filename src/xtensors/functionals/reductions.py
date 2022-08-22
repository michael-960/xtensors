from __future__ import annotations
from typing import Optional, overload
import numpy as np

from xtensors.functionals.base import DataArray, Functional
from xtensors import base
from xtensors.typing import NDArray, DimLike, DimsLike



class Reduction(Functional):
    '''
    Single-axis reduction fucntional
    '''
    def __init__(self, dim: DimLike) -> None:
        self.dim = dim
        self._reduce: base._reduc.ReductionFunc
        self.name = 'UNIMPLEMENTED_REDUCTION'

    def __call__(self, x: NDArray) -> DataArray:
        return self._reduce(x, self.dim)


class Mean(Reduction):
    def __init__(self, dim: int | str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanmean if nan else base.mean
        self.name = f'Mean({dim})'


class Sum(Reduction):
    def __init__(self, dim: int | str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nansum if nan else base.sum
        self.name = f'Sum({dim})'


class Std(Reduction):
    def __init__(self, dim: int | str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanstd if nan else base.std
        self.name = f'Std({dim})'

