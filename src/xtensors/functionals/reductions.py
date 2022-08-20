from __future__ import annotations
from typing import Optional, overload
import numpy as np
from numpy.typing import ArrayLike

from xtensors.functionals.base import DataArray, Functional
from xtensors import base



class Reduction(Functional):
    def __init__(self, dim: int|str) -> None:
        self.dim = dim
        self._reduce: base._reduc.ReductionFunc
        self.name = 'UNIMPLEMENTED_REDUCTION'


    def __call__(self, x: ArrayLike) -> DataArray:
        if isinstance(x, DataArray):
            return self._reduce(x, self.dim)
        else:
            assert isinstance(self.dim, int)
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

