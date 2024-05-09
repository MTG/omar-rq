from .classifier import Classifier

MODULES = {
    "classifier": Classifier,
}


def get_module(module_name: str):
    """Get module by name."""

    return MODULES[module_name]()
