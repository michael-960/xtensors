from ..typing import NDArray

from xarray import DataArray
import numpy as np

from ..tensor._base import XTensor

def generate_ignored_dims(n: int):
    return [f'__dim{i}' for i in range(n)]


def to_dataarray(x: NDArray) -> DataArray:
    if isinstance(x, DataArray): return x
    return DataArray(x, dims=generate_ignored_dims(len(x.shape)))


def to_xtensor(x: NDArray) -> XTensor:
    if isinstance(x, XTensor): return x

    return XTensor(x.__array__(), dims=generate_ignored_dims(len(x.shape)))


def to_ndarray(x: NDArray) -> np.ndarray:
    return x.__array__()
