from typing import TYPE_CHECKING as __TYPE_CHECKING

from ._typing import TensorLike, DimsLike, HasDimName, DimLike, Array, AxesPermutation, Dims, Coords

if __TYPE_CHECKING:
    from ._typing import AxesPermutation, BinaryOperator, TernaryOperator
    from ._typing import Function_2Args, Function_1Arg, Function_3Args
