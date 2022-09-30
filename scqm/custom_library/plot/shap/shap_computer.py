import torch
import random
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ShapComputer:
    def __init__(self, dataset, model, target_name, num_samples, max_display=10):
        self.dataset = dataset
        device = torch.device("cpu")
        self.model = model.to(device)
        self.target_name = target_name
        self.max_display = max_display
        # samples to compute contributions
        if target_name == "das283bsr_score":
            self.x_train = dataset.joint_das28_df_scaled_tensor_train[
                random.sample(
                    range(len(dataset.joint_das28_df_scaled_tensor_train)), num_samples
                )
            ].cpu()
            self.x_test = dataset.joint_das28_df_scaled_tensor_test.cpu()
            # for rescaling
            self.min_ = dataset.a_visit_df_scaling_values[0]["das283bsr_score"]
            self.max_ = dataset.a_visit_df_scaling_values[1]["das283bsr_score"]
            self.columns = dataset.joint_das28_df_columns_in_tensor
        else:
            self.x_train = dataset.joint_asdas_df_scaled_tensor_train[
                random.sample(
                    range(len(dataset.joint_asdas_df_scaled_tensor_train)), num_samples
                )
            ].cpu()
            self.x_test = dataset.joint_asdas_df_scaled_tensor_test.cpu()
            # for rescaling
            self.min_ = dataset.a_visit_df_scaling_values[0]["asdas_score"]
            self.max_ = dataset.a_visit_df_scaling_values[1]["asdas_score"]
            self.columns = dataset.joint_asdas_df_columns_in_tensor
        self.explainer = shap.Explainer(
            self.model_wrapper,
            pd.DataFrame(np.array(self.x_train), columns=self.columns),
        )
        self.shap_values = self.get_values(100)

    def model_wrapper(self, x):
        x_tensor = torch.tensor(x.values)
        return (
            self.model(x_tensor.float()).flatten() * (self.max_ - self.min_) + self.min_
        )

    def get_values(self, num_values):
        return self.explainer(
            pd.DataFrame(
                np.array(
                    self.x_test[random.sample(range(len(self.x_test)), num_values)]
                ),
                columns=self.columns,
            )
        )

    def summary_plot(self):
        plt.figure(figsize=(10, 10))
        shap.summary_plot(self.shap_values, max_display=self.max_display)

    def single_plot(self):
        plt.figure(figsize=(10, 10))
        shap.plots.waterfall(
            self.shap_values[random.choice(range(len(self.shap_values)))]
        )

    def force_plot(self):
        plt.figure(figsize=(10, 10))
        shap.plots.force(
            self.shap_values[random.choice(range(len(self.shap_values)))],
            matplotlib=True,
        )

    def bar_plot(self):
        plt.figure(figsize=(10, 10))
        shap.plots.bar(self.shap_values, max_display=self.max_display)

    def scatter_plot(self, feature_name="das283bsr_score"):
        plt.figure(figsize=(10, 10))
        shap.plots.scatter(self.shap_values[:, feature_name], color=self.shap_values)

    def heatmap(self):
        plt.figure(figsize=(10, 10))
        shap.plots.heatmap(self.shap_values)
