from collections import OrderedDict

from escnn import gspaces, nn
from gtc.utils import Config

from gtc.modules import FullyConnectedBlock, GonR2ConvBlock, GTtoT, Linear, Ravel
from gtc.pooling import GroupPooling

N = 8


conv1 = Config(
    {
        "type": GonR2ConvBlock,
        "params": {
            "N": N,
            "action": gspaces.flipRot2dOnR2,
            "n_channels": 20,
            "kernel_size": 16,
            "padding": 0,
            "bias": False,
        },
    }
)
gpool = Config(
    {
        "type": GroupPooling,
        "params": {},
    }
)
gttot = Config(
    {
        "type": GTtoT,
        "params": {},
    }
)
ravel = Config(
    {
        "type": Ravel,
        "params": {},
    }
)
FC1 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 350}})
FC2 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})
FC3 = Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}})
linear = Config({"type": Linear, "params": {"out_dim": 26}})
model_config = OrderedDict(
    {
        "conv1": conv1,
        "gpool": gpool,
        "gttot": gttot,
        "ravel": ravel,
        "FC1": FC1,
        "FC2": FC2,
        "FC3": FC3,
        "linear": linear,
    }
)
