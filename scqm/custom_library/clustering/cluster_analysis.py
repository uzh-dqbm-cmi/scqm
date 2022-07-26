import matplotlib.pyplot as plt
from scqm.custom_library.results.multiclass_results import MulticlassResults
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scqm.custom_library.clustering.utils import get_histories_and_features
import numpy as np
import pandas as pd
import seaborn as sns


class ClusterAnalysis:
    def __init__(self, dataset, model, trainer, n_clusters):
        self.dataset = dataset
        self.model = model
        self.trainer = trainer
        self.n_clusters = n_clusters
        (
            self.raw_features,
            self.raw_histories,
            self.raw_features_unscaled,
            self.raw_histories_unscaled,
            self.model_histories,
            self.subset_das28,
            self.subset_basdai,
        ) = get_histories_and_features(dataset, model, subset=dataset.train_ids)
        (
            self.raw_features_test,
            self.raw_histories_test,
            self.raw_features_unscaled_test,
            self.raw_histories_unscaled_test,
            self.model_histories_test,
            self.subset_das28_test,
            self.subset_basdai_test,
        ) = get_histories_and_features(dataset, model, subset=dataset.test_ids)
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=0, n_init=20, verbose=3
        ).fit(self.model_histories)
        self.clusters_test = self.kmeans.predict(self.model_histories_test)
        self.tsne_model = TSNE(n_components=2, random_state=0).fit_transform(
            self.model_histories_test
        )
        self.tsne_raw = TSNE(n_components=2, random_state=0).fit_transform(
            self.raw_histories_test
        )
        (
            self.feature_names,
            self.feature_continuous,
            self.feature_categorical,
        ) = self.get_feature_names()
        self.results = MulticlassResults(self.dataset, self.model, self.trainer)
        (
            self.df_das28,
            self.df_basdai,
            self.metrics_das28,
            self.metrics_basdai,
        ) = self.results.evaluate_model(self.dataset.test_ids)

    def get_feature_names(self):
        feature_names = []
        for event in self.dataset.event_names:
            feature_names.extend(list(getattr(self.dataset, event + "_df").columns))
        feature_names.extend(list(self.dataset.patients_df.columns))
        feature_continuous = []
        feature_categorical = []
        for index, name in enumerate(feature_names):
            try:
                self.raw_histories_unscaled_test[:, index].astype(float)
                feature_continuous.append(name)
            except (ValueError, TypeError) as e:
                tmp = [
                    item
                    for item in self.raw_histories_unscaled_test[:, index]
                    if item == item
                ]
                if len(set(tmp)) < 20:
                    feature_categorical.append(name)
        feature_continuous = [item for item in feature_continuous if item != "uid_num"]
        return feature_names, feature_continuous, feature_categorical

    def plot_clusters(self, raw=False):
        plt.figure(figsize=(10, 10))
        if raw:
            plt.scatter(
                self.tsne_raw[:, 0],
                self.tsne_raw[:, 1],
                c=self.clusters_test,
                alpha=0.4,
            )
            plt.title("Clusters on feature embeddings")
        else:
            plt.scatter(
                self.tsne_model[:, 0],
                self.tsne_model[:, 1],
                c=self.clusters_test,
                alpha=0.4,
            )
            plt.title("Clusters on model embeddings")
        return

    def plot_embeddings(self, raw=False):
        if raw:
            tsne_test = self.tsne_raw
        else:
            tsne_test = self.tsne_model
        for index, name in enumerate(self.feature_names):
            if name in self.feature_continuous:
                plt.figure(figsize=(10, 10))
                plt.scatter(
                    tsne_test[:, 0],
                    tsne_test[:, 1],
                    c=self.raw_histories_unscaled_test[:, index].astype(float),
                )
                plt.colorbar()
                plt.title(name)
            elif name in self.feature_categorical:
                tmp = [
                    item if item == item else np.nan
                    for item in self.raw_histories_unscaled_test[:, index]
                ]
                df = pd.DataFrame(
                    {"tsne_1": tsne_test[:, 0], "tsne_2": tsne_test[:, 1], "color": tmp}
                )
                plt.figure(figsize=(10, 10))
                colors = sns.color_palette("hls", len(set(df["color"].values)))
                for index, (i, dff) in enumerate(df.groupby("color")):
                    plt.scatter(
                        dff["tsne_1"],
                        dff["tsne_2"],
                        alpha=0.7,
                        label=i,
                        color=colors[index],
                    )
                    plt.title(name)
                plt.legend()
        return

    def plot_embeddings_versus_targets(self, raw=False):
        df_for_plot = pd.DataFrame(columns=list(self.df_das28.columns) + ["score"])
        for patient in self.subset_das28_test:
            tmp = self.df_das28[self.df_das28.patient_id == patient]
            tmp["score"] = "das28"
            df_for_plot = df_for_plot.append(tmp)
        for patient in self.subset_basdai_test:
            tmp = self.df_basdai[self.df_basdai.patient_id == patient]
            tmp["score"] = "basdai"
            df_for_plot = df_for_plot.append(tmp)

        plt.figure(figsize=(10, 10))
        tsne_test = self.tsne_raw if raw else self.tsne_model
        plt.scatter(tsne_test[:, 0], tsne_test[:, 1], c=df_for_plot.targets)
        plt.colorbar()
        plt.title("Targets")
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_test[:, 0], tsne_test[:, 1], c=df_for_plot.predictions)
        plt.colorbar()
        plt.title("Predictions")
        return

    # TODO implement was to change n_clusters easily
