import numpy as np

import random
import pickle
import io


class CPU_Unpickler(pickle.Unpickler):
    """Unpickles files saved using cpu or gpu"""

    def find_class(self, module, name):
        import torch

        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)

    # and then do :
    # with open(path, 'rb') as handle:
    #     #dataset = pickle.load(handle)
    #     file = CPU_Unpickler(handle).load()


def set_seeds(seed: int = 0, import_torch: bool = True) -> None:
    """set all relevant seeds

    Args:
        seed (int, optional): seed. Defaults to 0.
    """
    random.seed(seed)
    if import_torch:
        import torch

        torch.manual_seed(seed)
    np.random.seed(seed)
    return


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
