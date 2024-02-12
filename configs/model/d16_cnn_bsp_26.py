from collections import OrderedDict

from escnn import gspaces, nn
from gtc.utils import Config

from gtc.algebra import compute_non_redundant_tc_indices_dihedral
from gtc.modules import FullyConnectedBlock, GonR2ConvBlock, GTtoT, Linear, Ravel
from gtc.pooling import BspGroupPooling

N = 8

model_config = OrderedDict(
    {
        "conv1": Config(
            {
                "type": GonR2ConvBlock,
                "params": {
                    "N": N,
                    "action": gspaces.flipRot2dOnR2,
                    "n_channels": 20,
                    "kernel_size": 16,
                    "nonlinearity": None,
                    "padding": 0,
                    "bias": False,
                },
            }
        ),
        "gpool": Config(
            {
                "type": BspGroupPooling,
                "params": {"idx": None, "group_type": "dihedral"},
            }
        ),
        "ravel": Config(
            {
                "type": Ravel,
                "params": {},
            }
        ),
        "FC1": Config({"type": FullyConnectedBlock, "params": {"out_dim": 32}}),
        "FC2": Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}}),
        "FC3": Config({"type": FullyConnectedBlock, "params": {"out_dim": 64}}),
        "linear": Config({"type": Linear, "params": {"out_dim": 26}}),
    }
)
