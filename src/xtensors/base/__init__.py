
from ._misc import where, softmax, get_rank

from ._binary_op import (
    _add as add,
    _divide as divide,
    _multiply as multiply,

    _greater as greater,
    _greater_equal as greater_equal,
    _less as less,
    _less_equal as less_equal,
    _equal as equal,

    _or as logical_or,
    _and as logical_and,
)

from ._reduc import (
    _all as all,
    _any as any,
    _max as max,
    _min as min,
    _nanmax as nanmax,
    _nanmin as nanmin,

    _mean as mean,
    _nanmean as nanmean,
    _nanstd as nanstd,
    _nansum as nansum,
    _std as std,
    _sum as sum,
)


from ._arg import (
    _argmax as argmax,
    _argmin as argmin,
    _nanargmax as nanargmax,
    _nanargmin as nanargmin,
    _coordmax as coordmax,
    _coordmin as coordmin,
    _nancoordmax as nancoordmax,
    _nancoordmin as nancoordmin
)

from ._args import (argsmin, argsmax, nanargsmax, nanargsmin, ArgsFunction)

from ._coords import (coordsmin, coordsmax, nancoordsmax, nancoordsmin, CoordsFunction)


from ._reduc_2to1 import diagonal

from ._ufuncs import cos, cosh, exp, log, log2, log10, sigmoid, sin, sinh, tan, tanh 



