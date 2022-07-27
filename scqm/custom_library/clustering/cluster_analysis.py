import matplotlib.pyplot as plt
from scqm.custom_library.results.multiclass_results import MulticlassResults
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scqm.custom_library.clustering.utils import get_histories_and_features
import numpy as np
import pandas as pd
import seaborn as sns
import random
import torch


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
            self.patient_in_embedding,
            self.hist_per_event,
        ) = get_histories_and_features(dataset, model, subset=dataset.train_ids)
        (
            self.raw_features_test,
            self.raw_histories_test,
            self.raw_features_unscaled_test,
            self.raw_histories_unscaled_test,
            self.model_histories_test,
            self.subset_das28_test,
            self.subset_basdai_test,
            self.patient_in_embedding_test,
            self.hist_per_event_test,
        ) = get_histories_and_features(dataset, model, subset=dataset.test_ids)
        self.patient_info = {
            patient: {
                i: np.nan
                for i in range(len(self.patient_in_embedding[patient]["indices"]))
            }
            for patient in self.patient_in_embedding
        }
        self.patient_info_test = {
            patient: {
                i: np.nan
                for i in range(len(self.patient_in_embedding_test[patient]["indices"]))
            }
            for patient in self.patient_in_embedding_test
        }
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=0, n_init=20, verbose=3
        ).fit(self.model_histories)
        self.clusters_test = self.kmeans.predict(self.model_histories_test)
        self.clusters_train = self.kmeans.predict(self.model_histories)
        for patient in self.patient_info_test:
            for visit in self.patient_info_test[patient]:
                self.patient_info_test[patient][visit] = self.clusters_test[
                    self.patient_in_embedding_test[patient]["indices"][visit]
                ]
        self.tsne_model_all = TSNE(n_components=2, random_state=0).fit_transform(
            torch.cat((self.model_histories, self.model_histories_test))
        )
        self.tsne_raw_all = TSNE(n_components=2, random_state=0).fit_transform(
            torch.cat((self.raw_histories, self.raw_histories_test))
        )
        self.tsne_model_test = self.tsne_model_all[len(self.model_histories) :, :]
        self.tsne_raw_test = self.tsne_raw_all[len(self.model_histories) :, :]

        self.tsne_model_train = self.tsne_model_all[: len(self.model_histories), :]
        self.tsne_raw_train = self.tsne_raw_all[: len(self.model_histories), :]
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

    def get_history_for_event(self, event, subset_das28, subset_basdai, hist_per_event):
        numbers_of_target = [
            torch.sum(
                self.dataset.masks_das28.available_target_mask[
                    self.dataset.mapping_for_masks_das28[patient]
                ]
                == True
            ).item()
            for patient in subset_das28
        ]

        numbers_of_target.extend(
            [
                torch.sum(
                    self.dataset.masks_basdai.available_target_mask[
                        self.dataset.mapping_for_masks_basdai[patient]
                    ]
                    == True
                ).item()
                for patient in subset_basdai
            ]
        )
        event_histories = torch.empty(
            size=(sum(numbers_of_target), self.model.history_size)
        )
        index_in_tensor = 0
        for index, patient in enumerate(subset_das28):
            event_histories[
                index_in_tensor : index_in_tensor + numbers_of_target[index]
            ] = hist_per_event[patient][event]
            index_in_tensor += numbers_of_target[index]
        for index, patient in enumerate(subset_basdai):
            event_histories[
                index_in_tensor : index_in_tensor
                + numbers_of_target[index + len(subset_das28)]
            ] = hist_per_event[patient][event]
            index_in_tensor += numbers_of_target[index + len(subset_das28)]
        return event_histories

    def cluster_event(self, event="a_visit", n_clusters=2):
        event_histories = self.get_history_for_event(
            event, self.subset_das28, self.subset_basdai, self.hist_per_event
        )
        event_histories_test = self.get_history_for_event(
            event,
            self.subset_das28_test,
            self.subset_basdai_test,
            self.hist_per_event_test,
        )
        # clustering
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=0, n_init=20, verbose=3
        ).fit(event_histories)
        setattr(self, "kmeans_" + event, kmeans)
        setattr(self, "clusters_" + event + "_train", kmeans.predict(event_histories))
        setattr(
            self, "clusters_" + event + "_test", kmeans.predict(event_histories_test)
        )
        return

    def set_new_kmeans(self, n_clusters):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(
            n_clusters=self.n_clusters, random_state=0, n_init=20, verbose=3
        ).fit(self.model_histories)
        self.clusters_test = self.kmeans.predict(self.model_histories_test)
        self.clusters_train = self.kmeans.predict(self.model_histories)
        for patient in self.patient_info_test:
            for visit in self.patient_info_test[patient]:
                self.patient_info_test[patient][visit] = self.clusters_test[
                    self.patient_in_embedding_test[patient]["indices"][visit]
                ]

        return

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

    def plot_clusters(self, raw=False, subset="test"):
        plt.figure(figsize=(10, 10))
        if raw:
            if subset == "train":
                tsne = self.tsne_raw_train
                title = "Train clusters on feature embeddings"
                c = self.clusters_train
            else:
                tsne = self.tsne_raw_test
                title = "Test clusters on feature embeddings"
                c = self.clusters_test
        else:
            if subset == "train":
                tsne = self.tsne_model_train
                title = "Train clusters on model embeddings"
                c = self.clusters_train
            else:
                tsne = self.tsne_model_test
                title = "Test clusters on model embeddings"
                c = self.clusters_test

        plt.scatter(
            tsne[:, 0],
            tsne[:, 1],
            c=c,
            alpha=0.4,
        )
        plt.title(title)

        return

    def plot_embeddings(self, raw=False, subset="test"):
        if raw:
            if subset == "train":
                tsne = self.tsne_raw_train
                c = self.raw_histories_unscaled
            else:
                tsne = self.tsne_raw_test
                c = self.raw_histories_unscaled_test
        else:
            if subset == "train":
                tsne = self.tsne_model_train
                c = self.raw_histories_unscaled
            else:
                tsne = self.tsne_model_test
                c = self.raw_histories_unscaled_test
        for index, name in enumerate(self.feature_names):
            if name in self.feature_continuous:
                plt.figure(figsize=(10, 10))
                plt.scatter(
                    tsne[:, 0],
                    tsne[:, 1],
                    c=c[:, index].astype(float),
                )
                plt.colorbar()
                plt.title(name)
            elif name in self.feature_categorical:
                tmp = [item if item == item else np.nan for item in c[:, index]]
                df = pd.DataFrame(
                    {"tsne_1": tsne[:, 0], "tsne_2": tsne[:, 1], "color": tmp}
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
        tsne_test = self.tsne_raw_test if raw else self.tsne_model_test
        plt.scatter(tsne_test[:, 0], tsne_test[:, 1], c=df_for_plot.targets)
        plt.colorbar()
        plt.title("Targets")
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_test[:, 0], tsne_test[:, 1], c=df_for_plot.predictions)
        plt.colorbar()
        plt.title("Predictions")
        return

    def plot_patient_trajectory(self, patients=[], subset="test"):
        if len(patients) == 0:
            if subset == "test":
                patients = [
                    random.choice(self.subset_das28_test),
                    random.choice(self.subset_basdai_test),
                ]
            else:
                patients = [
                    random.choice(self.subset_das28),
                    random.choice(self.subset_basdai),
                ]
        tsne = self.tsne_model_train if subset == "train" else self.tsne_model_test
        embeddings = (
            self.patient_in_embedding
            if subset == "train"
            else self.patient_in_embedding_test
        )
        c = self.clusters_train if subset == "train" else self.clusters_test
        for patient in patients:
            plt.figure(figsize=(10, 10))
            plt.set_cmap("spring")
            plt.scatter(tsne[:, 0], tsne[:, 1], c=c, alpha=0.4)
            plt.scatter(
                tsne[embeddings[patient]["indices"], 0],
                tsne[embeddings[patient]["indices"], 1],
                alpha=1,
                s=70,
                color="black",
            )
            for index, elem in enumerate(
                range(1, len(embeddings[patient]["indices"]) + 1)
            ):
                plt.annotate(
                    elem,
                    (
                        tsne[embeddings[patient]["indices"], 0][index] + 2,
                        tsne[embeddings[patient]["indices"], 1][index] + 2,
                    ),
                )
        return

    # TODO method to highlight one patient, find if some patients move clusters (also potentially in train)
