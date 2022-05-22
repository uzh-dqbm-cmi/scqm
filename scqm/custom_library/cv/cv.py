from abc import ABC, abstractmethod
import imp

from scqm.custom_library.utils import set_seeds
from scqm.custom_library.partition.partition import DataPartition

class CV(ABC):
    def __init__(self, dataset, k):
        set_seeds(0)
        self.dataset = dataset
        self.k = k
        self.partition = DataPartition(self.dataset, k=self.k)

    def set_grid(self, parameters: dict):
        self.parameter_names = list(parameters.keys())
        self.parameters = parameters

    @abstractmethod
    def perform_cv(self):
        pass
