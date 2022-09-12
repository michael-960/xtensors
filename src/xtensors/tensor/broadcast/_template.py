'''
Template Broadcaster:

    First specify a template, then map axes of tensors onto the template

'''

from typing import Union
import numpy as np
from ._types import Broadcaster
from .._base import XTensor

from ..typing import Dims, Coords
from ..basic_utils import permute, permutation_well_defined



DimSelector = Union[str, int]


def template_broadcast(*dims: Union[str, int]) -> Broadcaster:
    selectors = []

    for d in dims:
        if isinstance(d, str):
            selectors.append(DimNameSelector(d))
    
    template = Template(*selectors)

    def _broadcast(X: XTensor, Y: XTensor):
        X1 = template.cast_and_update(X)
        Y1 = template.cast_and_update(Y)
        dims, coords = template.dims, template.coords
        template.clear()
        return X1.data, Y1.data, dims, coords

    return _broadcast




class AxisSelector:
    def __init__(self) -> None: ...

    def select_axis(self, X: XTensor) -> int|None: ...


class DimNameSelector(AxisSelector):
    def __init__(self, dimname: str, required: bool=False) -> None:
        self.dimname = dimname
        self.required = required

    def select_axis(self, X: XTensor) -> int|None:
        try:
            X.get_axis(self.dimname)
        except KeyError as e:
            if self.required: raise e
            return None


class Template:
    def __init__(self, *selectors: AxisSelector) -> None:
        self.selectors = selectors
        self._dims: Dims
        self._coords: Coords
        self.clear()
        self.check_well_defined()

    def check_well_defined(self):
        names = [sel.dimname for sel in self.selectors if isinstance(sel, DimNameSelector)]
        assert len(names) == len(set(names))

    def cast_and_update(self, X: XTensor) -> XTensor:
        axesp = [sel.select_axis(X) for sel in self.selectors]
        assert permutation_well_defined(axesp)
        return permute(X, axesp)

    def clear(self):
        self._dims = [None for _ in range(len(self.selectors))]
        self._coords= [None for _ in range(len(self.selectors))]

    @property
    def dims(self) -> Dims:
        return self._dims

    @property
    def coords(self) -> Coords:
        return self._coords
