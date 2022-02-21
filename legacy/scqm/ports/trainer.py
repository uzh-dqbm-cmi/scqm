from abc import ABC,abstractmethod
from .model import Model
from .domain import Datamanager

class Trainer(ABC):
    """

    """
    def __init__(self, model: Model, data: DataManager,config:dict):
        """
        :param model:
        :type model:
        :param config:
        :type config:
        """
        self.model = Model
        self.config = config
        self.data = DataManager

        pass

    @abstractmethod
    def train(self):