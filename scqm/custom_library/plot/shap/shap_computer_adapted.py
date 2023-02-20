import torch
import shap
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


class ShapComputerAdapted:
    def __init__(self, model, target_name, num_samples, columns, x_train, x_test, targ_max_, targ_min_, max_display=10):
        self.seed = 0
        device = torch.device("cpu")
        self.model = model.to(device)
        self.target_name = target_name
        self.max_display = max_display
        self.x_train = x_train[random.sample(range(len(x_train)), num_samples)].cpu()
        self.x_test = x_test.cpu()
        self.columns = columns
        self.targ_min_ = targ_min_
        self.targ_max_ = targ_max_
        self.explainer = shap.Explainer(
            self.model_wrapper,
            pd.DataFrame(np.array(self.x_train), columns=self.columns),
            seed=self.seed
        )
        self.shap_values = self.get_values(100)

    def model_wrapper(self, x):
        x_tensor = torch.tensor(x.values)
        return (
            self.model(x_tensor.float()).flatten().detach() * (self.targ_max_ - self.targ_min_) + self.targ_min_
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

    def scatter_plot(self, feature_name):
        plt.figure(figsize=(10, 10))
        shap.plots.scatter(self.shap_values[:, feature_name], color=self.shap_values)

    def heatmap(self):
        plt.figure(figsize=(10, 10))
        shap.plots.heatmap(self.shap_values)

    def custom_summary_plot(self):
        tmp = copy.copy(self.shap_values)
        tmp.values = np.delete(self.shap_values.values, list(self.columns).index(
            'Duration of morning stiffness: missing value'), axis=1)
        tmp.data = np.delete(self.shap_values.data, list(self.columns).index(
            'Duration of morning stiffness: missing value'), axis=1)
        shap.summary_plot(tmp, feature_names=[el for el in self.columns if el !=
                          'Duration of morning stiffness: missing value'], max_display=200)
