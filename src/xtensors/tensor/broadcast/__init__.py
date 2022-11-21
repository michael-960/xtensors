'''
1. Broadcasting
================

This module tries to address the problem of broadcasting two tensors.

Traditionally, two unnamed tensors are broadcast by aligning them from the right, e.g.

    X (64, 32, 8, 5)

    Y     (32, 8, 1)

are broadcastable, while

    X (64, 32, 8, 5)

    Y     (64, 32, 8)

are not.


The problem can be reduced to finding a consistent policy to match dimensions
of two given tensors. They are then broadcastable if and only if the matched
dimensions satisfy one of the following conditions:

    (a) they have the same length
    (b) one of them is singleton (length 1)


In general, two tensors are broadcastable if and only if:
    
    **there exists a matching scheme such that one of the tensors has its
    axes completely matched to those of the other one**


Formally, let X and Y be two tensors of shapes (x1, ..., xM) and (y1, ..., yN),
WLOG assume M > N, then they are broadcastable if and only if there exists a
matching

    (i1, j1), (i2, j2), ..., (iN, jN)

    such that (xip = yip) or (xip = 1) or (yip = 1)

    for p = 1, ..., N


2. Broadcasters
================

A *broadcaster* should:

    1. Permute the axes of X and Y, adding singleton dimensions if necessary,
       so that they are broadcastable by the default reverse-index policy

    2. 



3. Named Tensors
=================

This is further complicated when named dimensions are introduced. In general,
one may want to use named tensors without having to name all the dimensions,
consider for example two partially named tensors

    img: 

    .. code::

        shape  (10, 3,       256, 384)
        axis    0   1        2    3
        name    ?   CHANNEL  H    W


    lbl:

    .. code::

        shape  (10, 256, 384)
        axis    0   1    2
        name    ?   H    W

The desired broadcasting here should be 
    
    .. code::

        (img axis, lbl axis): (0, 0) (2, 1) (3, 2)

        output: (10, 3, 256, 384)



Another example:

    lbl_truth:

    .. code::

        shape   (20, 512, 512)
        axis     0   1    2
        name     ?   H    W

    lbl_prob:

    .. code::

        shape   (20, 3,     512, 512)
        axis     0   1      2    3
        name     ?   CLASS  H    W

    lbl_pred:

    .. code::

        shape   (20, 1,     17,    15,    512,  512)
        axis     ?   SCALE1 SCALE2 SCALE3 H     W


4. Principles
==============
    
    (a) The result of broadcasting two tensors X and Y should have the same
        number of dimensions as the tensor with more dimensions

    (b) If X and Y has dimensions with the same name, they should be broadcast
        together

'''

from ._broadcast import broadcast, vanilla_broadcaster, template_broadcaster, unilateral_broadcaster, cast
from ._dimcast import castdim, unilateral_dimcast
from ._template import Template, AxisSelector, IndexSelector, DimNameSelector

from typing import TYPE_CHECKING as __TYPE_CHECKING

from ._types import Broadcaster, Dimcaster, DimMerger, CoordMerger






