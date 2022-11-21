Argument Functions and Coordinate Functions
============================================

Argument functions, when fed with a single :py:class:`xtensors.TensorLike` object, return
an :py:class:`XTensor` instance with integer data representing the **indices**
of the original tensor. An example is :py:func:`xtensors.argsmax`, which returns
the indices where the maximum values occur:




.. math::

   x[i,j,k,l] 

.. math::

   y = \mathrm{argsmax}(x, \mathrm{dim}=1,2)

.. math::

   y[i,l,m] = \mathrm{argsmax}(x[i,:,:,l])[m]


Similarly, coordinate functions return the **coordinates** associated with the
axes of the original tensor.



.. autoclass:: xtensors.ArgsFunction
    :show-inheritance:

    .. automethod:: __call__

.. autoclass:: xtensors.CoordsFunction
    :show-inheritance:

    .. automethod:: __call__



.. autofunction:: xtensors.argsmax

.. autofunction:: xtensors.argsmin

.. autofunction:: xtensors.nanargsmax

.. autofunction:: xtensors.nanargsmin



.. autofunction:: xtensors.coordsmax

.. autofunction:: xtensors.coordsmin

.. autofunction:: xtensors.nancoordsmax

.. autofunction:: xtensors.nancoordsmin
