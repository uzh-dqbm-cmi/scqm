from scqm.custom_library.results.results import Results
from scqm.custom_library.metrics.metrics import Metrics
import torch
import pandas as pd


class BaselineResults(Results):
    def evaluate_model(self):
        if self.trainer.target_name == "das283bsr_score":
            test_tensor = self.dataset.joint_das28_df_scaled_tensor_test
            test_target = self.dataset.joint_targets_das28_df_scaled_tensor_test
            # scaling values
            min_ = self.dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
            max_ = self.dataset.a_visit_df_scaling_values[1]["das283bsr_score"]
        elif self.trainer.target_name == "asdas_score":
            test_tensor = self.dataset.joint_asdas_df_scaled_tensor_test
            test_target = self.dataset.joint_targets_asdas_df_scaled_tensor_test
            # scaling values
            min_ = self.dataset.a_visit_df_scaling_values[0]["asdas_score"]
            max_ = self.dataset.a_visit_df_scaling_values[1]["asdas_score"]

        with torch.no_grad():
            pred = self.model(test_tensor)
        # rescale
        pred_rescaled = pred * (max_ - min_) + min_
        target_rescaled = test_target * (max_ - min_) + min_
        # metrics
        mse = (
            sum((pred_rescaled - target_rescaled) ** 2) / len(target_rescaled)
        ).item()
        mae = (sum(abs(pred_rescaled - target_rescaled)) / len(target_rescaled)).item()

        print(f"MSE {mse} MAE {mae}")
        return
