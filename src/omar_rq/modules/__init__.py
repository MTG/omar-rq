from .classifier import Classifier
from .maskingmodel import MaskingModel
from .simclr import SimCLR

MODULES = {
    "classifier": Classifier,
    "simclr": SimCLR,
    "maskingmodel": MaskingModel,
}


def get_module(module_name: str):
    """Get module by name."""

    return MODULES[module_name]()
