from .base import where, argmax, argmin, softmax, get_rank

from .base import (diagonal, 
        sum, std, mean, 
        max, min, all, any, 
        nanmean, nansum, nanstd)


from .base import (
        exp, sigmoid, log, log10,
        cos, sin, tan,
        cosh, sinh, tanh
        )


from .base import (
        are_shapes_broadcastable, broadcast_shapes,
        are_xarrays_broadcastable, broadcast_xarrays, 
        are_arrays_broadcastable, broadcast_arrays
        )


from .base import (
        add, divide, 
        greater, greater_equal,
        less, less_equal,
        equal
        )


from .unify import (
        rename, 
        strip_dims, get_axis_from_dim, get_axes, 
        new_axes, new_axis)




#from .functionals.base import Pipe, Functional
