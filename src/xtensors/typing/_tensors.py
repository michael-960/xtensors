from __future__ import annotations
from typing import Union

import numpy as np

from numpy import typing as nptyping

import xarray as xr
import torch



NDArray = Union[nptyping.NDArray[np.float_], nptyping.NDArray[np.int_], nptyping.NDArray[np.bool_], xr.DataArray, torch.Tensor]

number = Union[int, float, complex, np.int_, np.float_, np.complex_, np.bool_]
Real: Union[int, float, np.int_, np.float_]


