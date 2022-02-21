from abc import ABC, abstractmethod
import os.path


class Metrics(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def plot(self) -> None:
        """
        Plot performance metrics.
        """
        pass

    @abstractmethod
    def print(self) -> None:
        """
        Print performance metrics.
        """
        pass

    @abstractmethod
    def save(self, filename: os.path = None) -> None:
        """
        Save performance metrics.
        """
        pass
