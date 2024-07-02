from .classifier import Classifier
from .simclr import SimCLR

MODULES = {
    "classifier": Classifier,
    "simclr": SimCLR,
}


def get_module(module_name: str):
    """Get module by name."""

    return MODULES[module_name]()
