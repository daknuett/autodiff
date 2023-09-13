from .autodiff import (
        gd, adam
        , Model
        , MatrixLayer, ReluLayer, BiasLayer
        , SequenceLayer, ParallelLayer, HeterogenousParallelLayer
        , IdentityLayer
        , Layer
        )

from .subsampling import SubsamplingLayer

from .util import get_dof
