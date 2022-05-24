import numpy as np
import pandas as pd
import torch
import numpy as np

from scqm.custom_library.metrics.metrics import Metrics
from scqm.custom_library.metrics.multiclass_metrics import MulticlassMetrics


class Results:
    # TODO implement naive baseline
    def __init__(self, dataset, model, trainer):
        self.dataset = dataset
        self.model = model
        self.trainer = trainer

    def evaluate_model(self, patient_ids):
        results_df = pd.DataFrame()
        for patient in patient_ids:
            # target_values = self.dataset[patient].targets_df['das283bsr_score'][self.dataset.min_num_visits - 1:].values
            # target_categories = self.dataset[patient].targets_df['das28_increase'][self.dataset.min_num_visits - 1:].values
            value_at_previous = (
                self.dataset[patient]
                .targets_df["das283bsr_score"][self.dataset.min_num_visits - 2 : -1]
                .values
            )

            (
                predictions,
                all_history,
                target_values,
                time_to_targets,
                target_categories,
            ) = self.model.apply(self.dataset, patient)

            results_df = results_df.append(
                pd.DataFrame(
                    {
                        "patient_id": patient,
                        "targets": target_values.flatten().cpu(),
                        "target_categories": target_categories.flatten().cpu(),
                        "predictions": predictions.flatten().cpu(),
                    }
                )
            )
        # self, device, possible_classes, predictions=None, true_values=None, predicted_probas=None
        if self.model.task == "classification":
            metrics = MulticlassMetrics(
                torch.device("cpu"),
                torch.tensor([0, 1, 2]),
                results_df["predictions"],
                results_df["target_categories"],
            )
            metrics_naive = None
            # naive : class p with probability preponderence of class p
            naive_predictions = np.random.choice(
                [0, 1, 2],
                size=len(results_df["target_categories"]),
                replace=True,
                p=np.array((1 / self.trainer.weights).cpu()),
            )
            # metrics_naive = MulticlassMetrics(torch.device('cpu'), torch.tensor([0, 1, 2]), pd.Series(naive_predictions),
            #                                   results_df['target_categories'])

        else:
            # rescale
            results_df["predictions"] = (
                results_df["predictions"]
                * (
                    self.dataset.a_visit_df_scaling_values[1]["das283bsr_score"]
                    - self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
                )
                + self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
            )
            results_df["targets"] = (
                results_df["targets"]
                * (
                    self.dataset.a_visit_df_scaling_values[1]["das283bsr_score"]
                    - self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
                )
                + self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
            )
            metrics = Metrics(
                torch.device("cpu"), results_df["predictions"], results_df["targets"]
            )
            # metrics_naive = Metrics(torch.device('cpu'), results_df['naive_base'], results_df['targets'])
        return results_df, metrics


def get_naive_baseline_regression(df):
    df["naive_base"] = [
        np.nan if index == 0 else df["targets"].iloc[index - 1]
        for index in range(len(df))
    ]
    return df
