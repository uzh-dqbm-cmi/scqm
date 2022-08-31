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
        results_df_asdas = pd.DataFrame()
        patients_das28 = [
            patient
            for patient in patient_ids
            if self.dataset[patient].target_name == "das283bsr_score"
        ]
        patients_asdas = [
            patient
            for patient in patient_ids
            if self.dataset[patient].target_name == "asdas_score"
        ]
        patients_both = [
            patient
            for patient in patient_ids
            if self.dataset[patient].target_name == "both"
        ]
        metrics_das28 = None
        metrics_asdas = None
        if len(patients_das28 + patients_both) > 0:
            for patient in patients_das28 + patients_both:
                (
                    predictions,
                    target_values,
                    time_to_targets,
                    prediction_dates,
                ) = self.model.apply(self.dataset, patient, "das283bsr_score")

                results_df_das28 = results_df_das28.append(
                    pd.DataFrame(
                        {
                            "patient_id": patient,
                            "targets": target_values.flatten().cpu(),
                            "predictions": predictions.flatten().cpu(),
                            "prediction_dates": prediction_dates,
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
        if len(patients_asdas + patients_both) > 0:
            for patient in patients_asdas + patients_both:
                (
                    predictions,
                    target_values,
                    time_to_targets,
                    prediction_dates,
                ) = self.model.apply(self.dataset, patient, "asdas_score")

                results_df_asdas = results_df_asdas.append(
                    pd.DataFrame(
                        {
                            "patient_id": patient,
                            "targets": target_values.flatten().cpu(),
                            "predictions": predictions.flatten().cpu(),
                            "prediction_dates": prediction_dates,
                        }
                    )
                )

            # rescale
            results_df_asdas["predictions"] = (
                results_df_asdas["predictions"]
                * (
                    self.dataset.a_visit_df_scaling_values[1]["asdas_score"]
                    - self.dataset.a_visit_df_scaling_values[0]["asdas_score"]
                )
                + self.dataset.a_visit_df_scaling_values[0]["asdas_score"]
            )
            results_df_asdas["targets"] = (
                results_df_asdas["targets"]
                * (
                    self.dataset.a_visit_df_scaling_values[1]["asdas_score"]
                    - self.dataset.a_visit_df_scaling_values[0]["asdas_score"]
                )
                + self.dataset.a_visit_df_scaling_values[0]["asdas_score"]
            )

            metrics_asdas = Metrics(
                torch.device("cpu"),
                results_df_asdas["predictions"],
                results_df_asdas["targets"],
            )
            # metrics_naive = Metrics(torch.device('cpu'), results_df['naive_base'], results_df['targets'])
        return results_df_das28, results_df_asdas, metrics_das28, metrics_asdas
