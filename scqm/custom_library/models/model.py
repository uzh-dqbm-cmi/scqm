from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self, config, device):
        self.config = config
        self.device = device

    @abstractmethod
    def train(self):
        pass

    def eval(self):
        raise NotImplementedError

    def apply(self):
        raise NotImplementedError
