from scqm.custom_library.results.results import Results
from scqm.custom_library.metrics.metrics import Metrics
import torch
import pandas as pd
import numpy as np


class BaselineResults(Results):
    def evaluate_model(self, subset, return_vectors=False):
        if self.trainer.target_name == "das283bsr_score":
            indices_features = np.concatenate(
                [
                    self.dataset.tensor_indices_mapping_test[patient]["joint_das28_df"]
                    for patient in subset
                ]
            )
            indices_targets = np.concatenate(
                [
                    self.dataset.tensor_indices_mapping_test[patient][
                        "joint_targets_das28_df"
                    ]
                    for patient in subset
                ]
            )
            test_tensor = self.dataset.joint_das28_df_scaled_tensor_test[
                indices_features
            ]
            test_target = self.dataset.joint_targets_das28_df_scaled_tensor_test[
                indices_targets,
                self.dataset.joint_targets_das28_df_columns_in_tensor.index("value"),
            ]
            # scaling values
            min_ = self.dataset.joint_das28_df_scaling_values[0]["das283bsr_score"]
            max_ = self.dataset.joint_das28_df_scaling_values[1]["das283bsr_score"]
            index_value_feature = self.dataset.joint_das28_df_columns_in_tensor.index(
                "das283bsr_score"
            )
        elif self.trainer.target_name == "asdas_score":
            indices_features = np.concatenate(
                [
                    self.dataset.tensor_indices_mapping_test[patient]["joint_asdas_df"]
                    for patient in subset
                ]
            )
            indices_targets = np.concatenate(
                [
                    self.dataset.tensor_indices_mapping_test[patient][
                        "joint_targets_asdas_df"
                    ]
                    for patient in subset
                ]
            )
            test_tensor = self.dataset.joint_asdas_df_scaled_tensor_test[
                indices_features
            ]
            test_target = self.dataset.joint_targets_asdas_df_scaled_tensor_test[
                indices_targets,
                self.dataset.joint_targets_asdas_df_columns_in_tensor.index("value"),
            ]
            # scaling values
            min_ = self.dataset.joint_asdas_df_scaling_values[0]["asdas_score"]
            max_ = self.dataset.joint_asdas_df_scaling_values[1]["asdas_score"]
            index_value_feature = self.dataset.joint_das28_df_columns_in_tensor.index(
                "asdas_score"
            )
        with torch.no_grad():
            pred = self.model(test_tensor).flatten()
        # rescale
        pred_rescaled = pred * (max_ - min_) + min_
        target_rescaled = test_target * (max_ - min_) + min_
        # for naive
        test_naive_rescaled = test_tensor[:, index_value_feature] * (max_ - min_) + min_
        # metrics
        mse = (
            sum((pred_rescaled - target_rescaled) ** 2) / len(target_rescaled)
        ).item()
        mae = (sum(abs(pred_rescaled - target_rescaled)) / len(target_rescaled)).item()

        print(f"MSE {mse} MAE {mae}")
        # naive mse
        naive_mse = (
            sum((test_naive_rescaled - target_rescaled) ** 2) / len(target_rescaled)
        ).item()
        print(f"naive mse {naive_mse}")
        if return_vectors:
            return mse, mae, naive_mse, pred_rescaled, target_rescaled
        else:
            return mse, mae, naive_mse
