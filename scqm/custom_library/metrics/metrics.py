import torch
import pandas as pd


class Metrics:
    """Base metric object"""

    def __init__(self, device: str, predictions=None, true_values=None):
        """Instantiate object

        Args:
            device (str): CPU or GPU
            predictions (_type_, optional): prediction array. Defaults to None.
            true_values (_type_, optional): true values array. Defaults to None.
        """
        if predictions is None:
            self.predictions = torch.empty(0, device=device)
            self.true_values = torch.empty(0, device=device)

        elif isinstance(predictions, pd.Series):
            self.predictions = torch.tensor(predictions.values)
            self.true_values = torch.tensor(true_values.values)

        else:
            self.predictions = predictions
            self.true_values = true_values

    def __len__(self) -> int:
        """Number of predictions/true values

        Raises:
            ValueError: if mismatch between number of predictions and true values

        Returns:
            int: number of preditions
        """
        if len(self.predictions) != len(self.true_values):
            raise ValueError("length of predictions doesnt match true values")
        else:
            return len(self.predictions)

    def add_observations(
        self, new_predictions: torch.tensor, new_true_values: torch.tensor
    ):
        """Include new predictions and corresponding true values

        Args:
            new_predictions (torch.tensor): new predictions
            new_true_values (torch.tensor): new true values
        """
        self.predictions = torch.cat([self.predictions, new_predictions])
        self.true_values = torch.cat([self.true_values, new_true_values])
        return

    def get_metrics(self, print_metric: bool = False):
        """Compute metric

        Args:
            print_metric (bool, optional): Print result. Defaults to False.
        """
        # mse
        self.returned_metric = (
            1 / len(self) * torch.sum((self.predictions - self.true_values) ** 2)
        )
        if print_metric:
            print(f"mse : {self.returned_metric}")
        return
