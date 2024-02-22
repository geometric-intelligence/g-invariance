import collections

from escnn import gspaces

import gtc.modules as gtc_modules
import gtc.pooling as gtc_pooling
from gtc.utils import Config


def insert_after_key(odict, after_key, key, value):
    """Insert an element into an OrderedDict after a specified key."""
    if after_key not in odict:
        raise KeyError(f"Key '{after_key}' not found in OrderedDict")

    temp_odict = collections.OrderedDict()
    inserted = False
    for k, v in odict.items():
        temp_odict[k] = v
        if k == after_key:
            temp_odict[key] = value
            inserted = True

    # If the specified key was the last one and the new element has been inserted,
    # it's already in the right place. Otherwise, raise an error.
    if not inserted:
        raise Exception("New element was not inserted. Check the 'after_key'.")

    return temp_odict


def get_model_config(group, pooling, n_filters, group_type="dihedral", out_dim=10):
    N = 8
    pooling_map = {
        "bsp": gtc_pooling.BspGroupPooling,
        "tc": gtc_pooling.TCGroupPooling,
        "max": gtc_pooling.GroupPooling,
    }
    model_config = collections.OrderedDict(
        {
            "conv1": Config(
                {
                    "type": gtc_modules.GonR2ConvBlock,
                    "params": {
                        "N": N,  # Assuming N is defined elsewhere
                        "action": gspaces.flipRot2dOnR2,  # Assuming gspaces is imported or defined elsewhere
                        "n_channels": 4,
                        "kernel_size": 16,
                        "padding": 0,
                        "bias": False,
                    },
                }
            ),
            "gpool": Config(
                {
                    "type": pooling_map[pooling],
                    "params": {"idx": None, "group_type": group_type},
                }
            ),
            "ravel": Config(
                {
                    "type": gtc_modules.Ravel,
                    "params": {},
                }
            ),
            "FC1": Config(
                {
                    "type": gtc_modules.FullyConnectedBlock,
                    "params": {"out_dim": out_dim},
                }
            ),
            "FC2": Config(
                {
                    "type": gtc_modules.FullyConnectedBlock,
                    "params": {"out_dim": out_dim},
                }
            ),
            "FC3": Config(
                {
                    "type": gtc_modules.FullyConnectedBlock,
                    "params": {"out_dim": out_dim},
                }
            ),
            "linear": Config(
                {
                    "type": gtc_modules.Linear,
                    "params": {"out_dim": out_dim},
                }
            ),
        }
    )

    if pooling == "max":
        gtot = Config(
            {
                "type": gtc_modules.GTtoT,
                "params": {},
            }
        )
        insert_after_key(model_config, after_key="gpool", key="gttot", value=gtot)

    return model_config
