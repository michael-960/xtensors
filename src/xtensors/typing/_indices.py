from __future__ import annotations
from typing import Tuple, Union




AxisDimPair = Tuple[int, str]
DimLike = Union[int, str, AxisDimPair]
DimsLike = Union[Tuple[int,...], Tuple[str,...], Tuple[AxisDimPair, ...]]
