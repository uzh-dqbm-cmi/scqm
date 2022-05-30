import torch
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from scqm.custom_library.metrics.metrics import Metrics


class BinaryMetrics(Metrics):
    """Metric for binary classification"""

    def __init__(
        self, device: str, predictions=None, true_values=None, predicted_probas=None
    ):
        """Instantiate object

        Args:
            device (str): CPU or GPU
            predictions (_type_, optional): prediction array. Defaults to None.
            true_values (_type_, optional): true values array. Defaults to None.
            predicted_probas (_type_, optional): predicted probabilities array. Defaults to None.
        """
        super().__init__(device, predictions, true_values)
        if predictions is None:
            self.predicted_probas = torch.empty(0, device=device)
        elif isinstance(predictions, pd.Series):
            self.predicted_probas = (
                torch.tensor(predicted_probas.values)
                if predicted_probas is not None
                else self.predictions
            )
        else:
            self.predicted_probas = (
                predicted_probas if predicted_probas else predictions
            )

    def discrete_metrics(self, print_confusion: bool = False):
        """Compute TN, FN, TP, FP, sensitivity, etc.

        Args:
            print_confusion (bool, optional): print confusion matrix. Defaults to False.

        Raises:
            ArithmeticError: if shape mismatch
        """
        self.TP = len(
            [
                elem
                for index, elem in enumerate(self.predictions)
                if elem == 1 and elem == self.true_values[index]
            ]
        )
        self.TN = len(
            [
                elem
                for index, elem in enumerate(self.predictions)
                if elem == 0 and elem == self.true_values[index]
            ]
        )
        self.FP = len(
            [
                elem
                for index, elem in enumerate(self.predictions)
                if elem == 1 and elem != self.true_values[index]
            ]
        )
        self.FN = len(
            [
                elem
                for index, elem in enumerate(self.predictions)
                if elem == 0 and elem != self.true_values[index]
            ]
        )
        if self.TP + self.TN + self.FP + self.FN != len(self):
            raise ArithmeticError("Sum of TP, TN, FP, FN doesnt match len of self")
        self.sensitivity = self.TP / (self.TP + self.FN)
        self.specificity = self.TN / (self.TN + self.FP)
        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.f1 = self.TP / (self.TP + 1 / 2 * (self.FP + self.FN))
        if print_confusion:
            print(
                pd.DataFrame(
                    data=[[self.TN, self.FN], [self.FP, self.TP]],
                    index=["pred 0", "pred 1"],
                    columns=["true 0", "true 1"],
                )
            )

    def get_auroc(self):
        fpr, tpr, thresholds = roc_curve(
            self.true_values, self.predicted_probas, pos_label=1
        )
        auc = roc_auc_score(self.true_values, self.predicted_probas)
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label=f"ROC curve (AUC = {np.round(auc,2)})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.show()
