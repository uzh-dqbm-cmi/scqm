from scqm.custom_library.results.results import Results
from scqm.custom_library.metrics.metrics import Metrics
import torch
import pandas as pd


class MulticlassResults(Results):
    def evaluate_model(self, patient_ids: list):
        """Apply model to patients in patient_ids

        Args:
            patient_ids (list): list of patients

        Returns:
            Tuple[pd.DataFrame, Metrics]: df with computed model predictions and metrics object for predictions
        """
        results_df_das28 = pd.DataFrame()
        results_df_basdai = pd.DataFrame()
        patients_das28 = [
            patient
            for patient in patient_ids
            if self.dataset[patient].target_name == "das283bsr_score"
        ]
        patients_basdai = [
            patient
            for patient in patient_ids
            if self.dataset[patient].target_name == "basdai_score"
        ]
        metrics_das28 = None
        metrics_basdai = None
        if len(patients_das28) > 0:
            for patient in patients_das28:
                (
                    predictions,
                    target_values,
                    time_to_targets,
                ) = self.model.apply(self.dataset, patient)

                results_df_das28 = results_df_das28.append(
                    pd.DataFrame(
                        {
                            "patient_id": patient,
                            "targets": target_values.flatten().cpu(),
                            "predictions": predictions.flatten().cpu(),
                        }
                    )
                )

            # rescale
            results_df_das28["predictions"] = (
                results_df_das28["predictions"]
                * (
                    self.dataset.a_visit_df_scaling_values[1]["das283bsr_score"]
                    - self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
                )
                + self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
            )
            results_df_das28["targets"] = (
                results_df_das28["targets"]
                * (
                    self.dataset.a_visit_df_scaling_values[1]["das283bsr_score"]
                    - self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
                )
                + self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
            )

            metrics_das28 = Metrics(
                torch.device("cpu"),
                results_df_das28["predictions"],
                results_df_das28["targets"],
            )
        if len(patients_basdai) > 0:
            for patient in patients_basdai:
                (
                    predictions,
                    target_values,
                    time_to_targets,
                ) = self.model.apply(self.dataset, patient)

                results_df_basdai = results_df_basdai.append(
                    pd.DataFrame(
                        {
                            "patient_id": patient,
                            "targets": target_values.flatten().cpu(),
                            "predictions": predictions.flatten().cpu(),
                        }
                    )
                )

            # rescale
            results_df_basdai["predictions"] = (
                results_df_basdai["predictions"]
                * (
                    self.dataset.basdai_df_scaling_values[1]["basdai_score"]
                    - self.dataset.basdai_df_scaling_values[0]["basdai_score"]
                )
                + self.dataset.basdai_df_scaling_values[0]["basdai_score"]
            )
            results_df_basdai["targets"] = (
                results_df_basdai["targets"]
                * (
                    self.dataset.basdai_df_scaling_values[1]["basdai_score"]
                    - self.dataset.basdai_df_scaling_values[0]["basdai_score"]
                )
                + self.dataset.basdai_df_scaling_values[0]["basdai_score"]
            )

            metrics_basdai = Metrics(
                torch.device("cpu"),
                results_df_basdai["predictions"],
                results_df_basdai["targets"],
            )
            # metrics_naive = Metrics(torch.device('cpu'), results_df['naive_base'], results_df['targets'])
        return results_df_das28, results_df_basdai, metrics_das28, metrics_basdai