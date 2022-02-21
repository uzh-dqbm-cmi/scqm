"""
Model interface
"""
from .metrics import Metrics

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import os.path


class Model(ABC):
    """
    Abstract model class that acts as an interface for training and evaluating machine learning models.
    Specific models implement this class in the adapter. The objective is to remove the dependency on frameworks such as
    sklearn, pytorch, tensorflow... and keep the code clean.
    """

    def __init__(self, config: dict):
        """
        Initialize a new model object, according to the configurations passed in

        Args:
            config: dictionary containing the configuration parameters specific to the model
        """
        # self.config = config
        pass

    @abstractmethod
    def train(self, train_data: Tuple[np.array, np.array], val_data: Tuple[np.array, np.array]) -> None:
        """
        Train the model using the train and validation data passed in as two tuples.

        Args:
            train_data (tuple): the training data to be used for the model (first element: features array of shape
            num_samples x num_features, second element: target array of shape num_samples x 1)
            val_data (tuple): the validation data to be used for the model (first element: features array of shape
            num_samples x num_features, second element: target array of shape num_samples x 1)
        """
        pass

    @abstractmethod
    def test(self, test_data: Tuple[np.array, np.array]) -> None:
        """
        Test the model using the test data and print out specific test metrics.

        Args:
            test_data (tuple): the test data to be used for the model (first element: features array of shape
            num_samples x num_features, second element: target array of shape num_samples x 1)
        """
        pass

    @abstractmethod
    def predict(self, x: np.array) -> np.array:
        """
        Perform predictions with the model on specific data

        Args:
            x: Input features to be used to perform prediction using the model. The data needs to be formatted to shape
            num_samples x num_features.

        Returns:
            y (np.array): array containing the predictions for the passed in data (num_samples x 1)
        """
        pass

    @abstractmethod
    def save(self, filename: os.path) -> os.path:
        """
        Save the model in a joblib.pkl file.
        Args:
            filename: Filename used to store the model. If None is provided, a filename of the following pattern will be
                      generated: <uuid>_<model_name>.joblib.pkl

        Returns:
            os.path: Filename tht was used to store the model.
        """
        pass

    @abstractmethod
    def metrics(self) -> Metrics:
        """
        Get all relevant performance metrics of the model.
        Returns:
            Metrics: Metrics object specified in Metrics class.
        """
        pass
