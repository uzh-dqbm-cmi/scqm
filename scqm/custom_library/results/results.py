from typing import Tuple
import numpy as np
import pandas as pd
import torch
import numpy as np

from scqm.custom_library.metrics.metrics import Metrics
from scqm.custom_library.metrics.multiclass_metrics import MulticlassMetrics
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.models.model import Model
from scqm.custom_library.trainers.trainer import Trainer


class Results:
    """Base class to handle model results"""

    def __init__(self, dataset, model, trainer):

        self.dataset = dataset
        self.model = model
        self.trainer = trainer

    def evaluate_model(self, patient_ids: list):
        """Apply model to patients in patient_ids

        Args:
            patient_ids (list): list of patients

        Returns:
            Tuple[pd.DataFrame, Metrics]: df with computed model predictions and metrics object for predictions
        """
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
                visit_ids,
            ) = self.model.apply(self.dataset, patient)
            prev_visit_dates = []
            prev_visit_ids = []
            for index, elem in enumerate(self.dataset[patient].visit_ids):
                if elem in visit_ids:
                    index_prev = index - 1

                    while (
                        index_prev >= 0
                        and (
                            pd.Timestamp(self.dataset[patient].visits[index].date)
                            - pd.Timestamp(
                                self.dataset[patient].visits[index_prev].date
                            )
                        ).days
                        < self.dataset.masks.min_time_since_last_event
                    ):
                        index_prev -= 1
                    if index_prev >= 0:

                        prev_visit_dates.append(
                            self.dataset[patient].visits[index_prev].date
                        )
                        prev_visit_ids.append(
                            self.dataset[patient].visits[index_prev].id
                        )
                    else:
                        prev_visit_dates.append(np.nan)
                        prev_visit_ids.append(np.nan)

            # prev_visit_dates = [self.dataset[patient].visits[index - 1].date for index, elem in enumerate(
            #     self.dataset[patient].visit_ids) if elem in visit_ids]
            # prev_visit_ids = [self.dataset[patient].visits[index - 1].id for index,
            #                   elem in enumerate(self.dataset[patient].visit_ids) if elem in visit_ids]
            pred_dates = [
                self.dataset[patient].visits[index].date
                for index, elem in enumerate(self.dataset[patient].visit_ids)
                if elem in visit_ids
            ]

            # results_df = results_df.append(
            #     pd.DataFrame(
            #         {
            #             "patient_id": patient,
            #             "targets": target_values.flatten().cpu(),
            #             "target_categories": target_categories.flatten().cpu(),
            #             "predictions": predictions.flatten().cpu(),
            #             "time_to_predictions": time_to_targets.flatten().cpu(),
            #             "visit_ids": visit_ids
            #         }
            #     )
            # )
            results_df = results_df.append(
                pd.DataFrame(
                    {
                        "patient_id": patient,
                        "targets": target_values.flatten().cpu(),
                        "target_categories": target_categories.flatten().cpu(),
                        "predictions": predictions.flatten().cpu(),
                        "date": pred_dates,
                        "prev_visit_date": prev_visit_dates,
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
            # results_df["time_to_predictions"] = (results_df["time_to_predictions"]
            #                                  * (
            #     self.dataset.targets_df_scaling_values[1]["date"]
            #     - self.dataset.targets_df_scaling_values[0]["date"]
            # )
            #     + self.dataset.targets_df_scaling_values[0]["date"])
            metrics = Metrics(
                torch.device("cpu"), results_df["predictions"], results_df["targets"]
            )
            # metrics_naive = Metrics(torch.device('cpu'), results_df['naive_base'], results_df['targets'])
        return results_df, metrics

    def post_process(self, df):
        df_post = df.copy()

        def num_visits(df):
            df["num_visits"] = [elem + 1 for elem in range(len(df))]
            df["avail_visits"] = len(df)
            return df

        def posthoc(df):
            patient = df["patient_id"].iloc[0]
            target_df = self.dataset[patient].targets_df.copy()
            das28_prev = []
            for date_prev, date in zip(df["prev_visit_date"], df["date"]):
                das28_prev.append(
                    target_df[target_df.date == date_prev].das283bsr_score.item()
                )
            #         das28_true = target_df[target_df.date == date].das283bsr_score.item()
            #         # das28_true = df[df.date == date].targets.item()
            #         das28_pred = df[df.date == date].predictions.item()
            df["prev_value"] = das28_prev
            return df

        df_post["days_to_prev"] = (df_post["date"] - df_post["prev_visit_date"]).apply(
            lambda x: x.days
        )
        df_post = df_post.groupby("patient_id").apply(num_visits)
        df_post["ae"] = abs(df_post["predictions"] - df_post["targets"])
        df_post = (
            df_post.dropna()
            .groupby("patient_id")
            .apply(posthoc, self.dataset)
            .drop(columns=["target_categories"])
        )
        df_post["true_cat"] = [
            0
            if (
                abs(df_post.iloc[index].targets - df_post.iloc[index].prev_value) <= 1.2
            )
            else 1
            if (df_post.iloc[index].targets > df_post.iloc[index].prev_value)
            else 2
            for index in range(len(df_post))
        ]
        df_post["pred_cat"] = [
            0
            if (
                abs(df_post.iloc[index].predictions - df_post.iloc[index].prev_value)
                <= 1.2
            )
            else 1
            if (df_post.iloc[index].predictions > df_post.iloc[index].prev_value)
            else 2
            for index in range(len(df_post))
        ]
        return df_post

    def get_naive_baseline_regression(df):
        df["naive_base"] = [
            np.nan if index == 0 else df["targets"].iloc[index - 1]
            for index in range(len(df))
        ]
        return df
