from abc import ABC, abstractmethod

# TODO in model trainings, rename indices, index, combined, combined lstm etc... its a mess
class Model(ABC):
    """
    Base model class"""

    def __init__(self, config: dict, device: str):
        """Instantiate model

        Args:
            config (dict): model parameters
            device (str): device
        """
        self.config = config
        self.device = device

    def train(self):
        self.train()

    def eval(self):
        self.eval()

    def apply(self):
        raise NotImplementedError
