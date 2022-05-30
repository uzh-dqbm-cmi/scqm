from abc import ABC, abstractmethod


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

    @abstractmethod
    def train(self):
        pass

    def eval(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError
