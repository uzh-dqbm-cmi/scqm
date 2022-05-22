import numpy as np
import torch
import random
import pickle
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)

    # and then do :
    # with open(path, 'rb') as handle:
    #     #dataset = pickle.load(handle)
    #     file = CPU_Unpickler(handle).load()


def set_seeds(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return

