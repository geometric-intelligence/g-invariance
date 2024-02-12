from collections import OrderedDict

from escnn import gspaces, nn
from gtc.utils import Config

from gtc.algebra import compute_non_redundant_tc_indices_cyclic
from gtc.modules import (
    BatchNorm1D,
    FullyConnectedBlock,
    GonR2ConvBlock,
    GTtoT,
    Linear,
    Ravel,
)
from gtc.pooling import TCGroupPooling

N = 8


"""
CONV 1
"""
conv1 = Config(
    {
        "type": GonR2ConvBlock,
        "params": {
            "N": N,
            "action": gspaces.rot2dOnR2,
            "nonlinearity": None,
            "n_channels": 24,
            "kernel_size": 16,
            "padding": 0,
            "bias": False,
        },
    }
)


"""
GROUP POOL
"""

gpool = Config(
    {
        "type": TCGroupPooling,
        "params": {
            "idx": compute_non_redundant_tc_indices_cyclic(N=N),
            "group_type": "cyclic",
        },
    }
)


"""
RAVEL
"""

ravel = Config(
    {
        "type": Ravel,
        "params": {},
    }
)


"""
FC1
"""

FC1 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})


"""
FC2
"""

FC2 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})


"""
FC3
"""

FC3 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})


"""
LINEAR
"""
linear = Config({"type": Linear, "params": {"out_dim": 26}})


"""
MODEL CONFIG
"""

model_config = OrderedDict(
    {
        "conv1": conv1,
        "gpool": gpool,
        "ravel": ravel,
        "FC1": FC1,
        "FC2": FC2,
        "FC3": FC3,
        "linear": linear,
    }
)
