import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import torch.nn.functional as F

from scqm.custom_library.metrics.metrics import Metrics


class MulticlassMetrics(Metrics):
    """Metrics for multiclass classification problem"""

    def __init__(
        self,
        device: str,
        possible_classes,
        predictions=None,
        true_values=None,
        predicted_probas=None,
    ):
        """Instantiate object

        Args:
            device (str): CPU or GPU
            possible_classes (_type_): possible classes
            predictions (_type_, optional): array of predictions. Defaults to None.
            true_values (_type_, optional): array of true values. Defaults to None.
            predicted_probas (_type_, optional): array of predicted probabilities. Defaults to None.
        """
        super().__init__(device, predictions, true_values)
        self.possible_classes = possible_classes
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

    def get_metrics(self, print_metrics: str = False):
        """Compute multiclass metrics

        Args:
            print_metrics (str, optional): Print the result. Defaults to False.
        """
        # macro f1
        self.returned_metric = 0
        self.fpr = np.empty(len(self.possible_classes))
        self.tpr = np.empty(len(self.possible_classes))
        self.all_metrics = []
        for class_ in self.possible_classes:
            TP = len(
                [
                    elem
                    for index, elem in enumerate(self.predictions)
                    if elem == class_ and elem == self.true_values[index]
                ]
            )
            FP = len(
                [
                    elem
                    for index, elem in enumerate(self.predictions)
                    if elem == class_ and elem != self.true_values[index]
                ]
            )
            FN = len(
                [
                    elem
                    for index, elem in enumerate(self.predictions)
                    if elem != class_ and class_ == self.true_values[index]
                ]
            )
            TN = len(
                [
                    elem
                    for index, elem in enumerate(self.predictions)
                    if elem != class_ and class_ != self.true_values[index]
                ]
            )
            if TP + FN != len([elem for elem in self.true_values if elem == class_]):
                raise (ArithmeticError("number of positives dont match"))
            if TP + FP == 0:
                precision = 0
            else:
                precision = TP / (TP + FP)
            if TP + FN == 0:
                recall = 0
            else:
                recall = TP / (TP + FN)
            if precision + recall == 0:
                print(f"Setting f1 to zero for class {class_} because division by zero")
                F1 = 0
            else:
                F1 = 2 * (precision * recall) / (precision + recall)
            self.all_metrics.append(F1)
            self.returned_metric += F1
            self.fpr[class_] = FP / (FP + TN)
            self.tpr[class_] = TP / (TP + FN)
        self.returned_metric = self.returned_metric / len(self.possible_classes)
        if print_metrics:
            print(f"macro f1 {self.returned_metric} all {self.all_metrics}")
        return

    def get_auroc(self):
        """compute auroc"""
        true_values_one_hot = F.one_hot(self.true_values)
        plt.figure()
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        for class_ in self.possible_classes:
            fpr, tpr, thresholds = roc_curve(
                true_values_one_hot[:, class_],
                self.predicted_probas[:, class_],
                pos_label=1,
            )
            # auc = roc_auc_score(self.true_values, self.predicted_probas)
            lw = 2
            plt.plot(
                fpr,
                tpr,
                lw=lw,
                label=f"ROC curve class {class_}",
            )
        plt.show()
