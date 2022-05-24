import torch
import pandas as pd


class Metrics:
    def __init__(self, device, predictions=None, true_values=None):
        if predictions is None:
            self.predictions = torch.empty(0, device=device)
            self.true_values = torch.empty(0, device=device)

        elif isinstance(predictions, pd.Series):
            self.predictions = torch.tensor(predictions.values)
            self.true_values = torch.tensor(true_values.values)

        else:
            self.predictions = predictions
            self.true_values = true_values

    def __len__(self):
        if len(self.predictions) != len(self.true_values):
            raise ValueError("length of predictions doesnt match true values")
        else:
            return len(self.predictions)

    def add_observations(self, new_predictions, new_true_values):
        self.predictions = torch.cat([self.predictions, new_predictions])
        self.true_values = torch.cat([self.true_values, new_true_values])
        return

    def get_metrics(self, print_metric=False):
        # mse
        self.returned_metric = (
            1 / len(self) * torch.sum((self.predictions - self.true_values) ** 2)
        )
        if print_metric:
            print(f"mse : {self.returned_metric}")
        return
