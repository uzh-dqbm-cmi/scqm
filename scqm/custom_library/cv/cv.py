from abc import ABC, abstractmethod

from scqm.custom_library.utils import set_seeds
from scqm.custom_library.partition.partition import DataPartition


class CV(ABC):
    """
    Base cross validation object. Create partition on data.
    """

    def __init__(self, dataset, k: int = 5) -> None:
        """Init method. Create the partition on the data.

        Args:
            dataset (Dataset): dataset
            k (int): Number of folds
        """
        set_seeds(0)
        self.dataset = dataset
        self.k = k
        self.partition = DataPartition(self.dataset, k=self.k)

    @abstractmethod
    def perform_cv(self):
        pass
