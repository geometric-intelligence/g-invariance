from gtc.utils import Config
from gtc.utils import WBLogger

"""
LOGGING
"""
logger_config = Config(
    {
        "type": WBLogger,
        "params": {
            "project": "bispectrum",
            "data_project": "bispectrum",
            "entity": "johan-atmo",
            "log_interval": 1,
            "watch_interval": 1,
            "plot_interval": 1,
            "end_plotter": None,
            "step_plotter": None,
        },
    }
)
