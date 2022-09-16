from __future__ import annotations

from xtensors.functionals.base import Functional
from xtensors import base

from .. import tensor as xtt


class Reduction(Functional):
    '''
    Single-axis reduction fucntional
    '''
    def __init__(self, dim: xtt.DimLike, /, *args) -> None:
        self.dim = dim
        self._reduce: base._reduc.ReductionFunc | base._arg.ArgFunction
        self.name = 'UNIMPLEMENTED_REDUCTION'

    def __call__(self, x: xtt.TensorLike) -> xtt.XTensor:
        return self._reduce(x, self.dim)


class Index(Reduction):
    def __init__(self, dim: xtt.DimLike, index: int):
        self.dim = dim
        self.index = index
        self.name = f'Index({dim},{index})'

    @xtt.generalize_at_1
    def __call__(self, X: xtt.XTensor) -> xtt.XTensor:
        return xtt.index(X, (self.dim, self.index))


class Mean(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanmean if nan else base.mean
        self.name = f'Mean({dim})'


class Sum(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nansum if nan else base.sum
        self.name = f'Sum({dim})'


class Std(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanstd if nan else base.std
        self.name = f'Std({dim})'


class Max(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanmax if nan else base.max
        self.name = f'Max({dim})'


class Min(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanmin if nan else base.min
        self.name = f'Max({dim})'


class ArgMax(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanargmax if nan else base.argmax
        self.name = f'ArgMax({dim})'


class ArgMin(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nanargmin if nan else base.argmin
        self.name = f'ArgMin({dim})'


class CoordMax(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nancoordmax if nan else base.coordmax
        self.name = f'CoordMax({dim})'


class CoordMin(Reduction):
    def __init__(self, dim: int|str, nan: bool=True) -> None:
        super().__init__(dim)
        self._reduce = base.nancoordmin if nan else base.coordmin
        self.name = f'CoordMin({dim})'
