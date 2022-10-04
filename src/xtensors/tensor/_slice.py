from __future__ import annotations
from abc import abstractmethod

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from typing import Union
    from ._base import XTensor



class MetaTensorSlice:
    def __init__(self, dimname: str):
        self._dimname = dimname

    def __getitem__(self, slc: Union[int, slice]) -> TensorIndexer:
        if isinstance(slc, slice):
            return TensorSlice(self.dimname, slc.start, slc.stop, slc.step)
        else:
            return SingleIndex(self.dimname, slc)
        
    @property
    def dimname(self) -> str: return self._dimname

    def __get_dimname__(self) -> str:
        return self.dimname



class TensorIndexer:
    @abstractmethod
    def index(self, X: XTensor) -> XTensor: ...


class TensorSlice(TensorIndexer):
    def __init__(self, dimname: str, start, stop, step):
        self._dimname = dimname
        self._slice = slice(start, stop, step)

    def index(self, X: XTensor) -> XTensor:
        Y = X.viewcopy()
        axis = Y.get_axis(self._dimname)

        coord = X.get_coord(axis)
        coord = None if coord is None else coord[self._slice]

        slices = [slice(None, None, None) for _ in range(X.rank)]
        slices[axis] = self._slice

        Y.data = X.data[tuple(slices)]

        Y.set_coord(axis, coord=coord)
        return Y


class SingleIndex(TensorIndexer):
    def __init__(self, dimname: str, ind: int):
        self._dimname = dimname
        self.ind = ind

    def index(self, X: XTensor) -> XTensor:
        return X.get(self._dimname, self.ind)



