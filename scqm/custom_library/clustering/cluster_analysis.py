import matplotlib.pyplot as plt
from scqm.custom_library.results.multiclass_results import MulticlassResults
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scqm.custom_library.clustering.utils import get_histories_and_features
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
import numpy as np
import pandas as pd
import seaborn as sns
import random
import torch
from yellowbrick.cluster import KElbowVisualizer
from tqdm import tqdm
from scqm.custom_library.clustering.similarity import compute_similarity


class ClusterAnalysis:
    def __init__(self, dataset, model, trainer, n_clusters):
        self.dataset = dataset
        self.model = model
        self.trainer = trainer
        self.n_clusters = n_clusters
        (
            self.raw_features,
            self.raw_histories,
            self.raw_features_all,
            self.raw_features_unscaled,
            self.raw_histories_unscaled,
            self.model_histories,
            self.subset_das28,
            self.subset_basdai,
            self.patient_in_embedding,
            self.hist_per_event,
        ) = get_histories_and_features(dataset, self.model, subset=dataset.train_ids)
        (
            self.raw_features_test,
            self.raw_histories_test,
            self.raw_features_all_test,
            self.raw_features_unscaled_test,
            self.raw_histories_unscaled_test,
            self.model_histories_test,
            self.subset_das28_test,
            self.subset_basdai_test,
            self.patient_in_embedding_test,
            self.hist_per_event_test,
        ) = get_histories_and_features(dataset, self.model, subset=dataset.test_ids)
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
        for patient in self.patient_info:
            for visit in self.patient_info[patient]:
                self.patient_info[patient][visit] = self.clusters_train[
                    self.patient_in_embedding[patient]["indices"][visit]
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
        self.get_diagnosis()

    def get_diagnosis(self):
        df_dict = load_dfs_all_data()
        table_patients = df_dict["patients"]
        types = [
            "diagnose_rheumatoid_arthritis",
            "diagnose_ankylosing_spondylitis",
            "diagnose_psoriasis_arthritis",
            "diagnose_undifferentiated_arthritis",
        ]
        patients_per_type = list()
        for tp in types:
            patients_per_type.append(
                table_patients[table_patients[tp] == "yes"]["patient_id"].unique()
            )
        for patient in self.patient_in_embedding:
            for index, elem in enumerate(types):
                if patient in patients_per_type[index]:
                    self.patient_in_embedding[patient]["disease"] = elem
        for patient in self.patient_in_embedding_test:
            for index, elem in enumerate(types):
                if patient in patients_per_type[index]:
                    self.patient_in_embedding_test[patient]["disease"] = elem
        return

    def find_cluster_number(self, event=None):
        if event is not None:
            x = getattr(self, event + "_histories_train")
            title = event
        else:
            x = self.model_histories
            title = "all"
        plt.figure(figsize=(7, 7))
        visualizer = KElbowVisualizer(KMeans(), k=(2, 8))
        plt.title(title)
        visualizer.fit(np.array(x.cpu()))
        plt.figure(figsize=(7, 7))
        visualizer = KElbowVisualizer(KMeans(), k=(2, 8), metric="calinski_harabasz")
        visualizer.fit(np.array(x.cpu()))
        plt.title(title)
        return

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

    def tsne_for_event(self, event="a_visit"):
        event_histories = self.get_history_for_event(
            event, self.subset_das28, self.subset_basdai, self.hist_per_event
        )
        event_histories_test = self.get_history_for_event(
            event,
            self.subset_das28_test,
            self.subset_basdai_test,
            self.hist_per_event_test,
        )
        setattr(self, event + "_histories_train", event_histories)
        setattr(self, event + "_histories_test", event_histories_test)
        tsne_event = TSNE(n_components=2, random_state=0).fit_transform(
            torch.cat((event_histories, event_histories_test))
        )
        tsne_train = tsne_event[: len(event_histories), :]
        tsne_test = tsne_event[len(event_histories) :, :]

        setattr(self, "tsne_" + event, tsne_event)
        setattr(self, "tsne_train_" + event, tsne_train)
        setattr(self, "tsne_test_" + event, tsne_test)

        return

    def plot_embeddings_event_cluster(self, n_clusters, event="a_visit", subset="test"):
        # clustering
        event_histories = getattr(self, event + "_histories_train")
        event_histories_test = getattr(self, event + "_histories_test")

        kmeans = KMeans(
            n_clusters=n_clusters, random_state=0, n_init=20, verbose=3
        ).fit(event_histories)
        clusters_train = kmeans.predict(event_histories)
        clusters_test = kmeans.predict(event_histories_test)
        setattr(self, "kmeans_" + event, kmeans)
        setattr(self, "clusters_" + event + "_train", clusters_train)
        setattr(self, "clusters_" + event + "_test", clusters_test)

        if subset == "train":
            tsne = getattr(self, "tsne_train_" + event)
            c = self.raw_histories_unscaled
        else:
            tsne = getattr(self, "tsne_test_" + event)
            c = self.raw_histories_unscaled_test
            # plot
        plt.figure(figsize=(10, 10))
        plt.scatter(
            tsne[:, 0],
            tsne[:, 1],
            c=getattr(self, "clusters_" + event + "_" + subset),
            alpha=0.4,
        )
        plt.title(event + " clusters")
        for index, name in enumerate(self.feature_names):
            if name in self.feature_continuous:
                plt.figure(figsize=(10, 10))
                plt.scatter(
                    tsne[:, 0],
                    tsne[:, 1],
                    c=c[:, index].astype(float),
                )
                plt.colorbar()
                plt.title(event + " decomposition : " + name)
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
                    plt.title(event + " decomposition : " + name)
                plt.legend()
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
        for patient in self.patient_info:
            for visit in self.patient_info[patient]:
                self.patient_info[patient][visit] = self.clusters_train[
                    self.patient_in_embedding[patient]["indices"][visit]
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
        c = [
            index + 1
            for patient in self.patient_in_embedding_test
            for index in range(len(self.patient_in_embedding_test[patient]["indices"]))
        ]
        plt.figure(figsize=(10, 10))
        plt.scatter(
            self.tsne_model_test[:, 0], self.tsne_model_test[:, 1], c=c, alpha=0.4
        )
        plt.colorbar()
        plt.title("Number of predictions")
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

    def plot_diagnoses(self, raw=False, subset="test"):
        if subset == "test":
            embeddings = self.patient_in_embedding_test
            tsne = self.tsne_raw_test if raw else self.tsne_model_test
        else:
            embeddings = self.patient_in_embedding
            tsne = self.tsne_raw_train if raw else self.tsne_model_train
        c = []
        for patient in embeddings:
            for index in range(len(embeddings[patient]["indices"])):
                c.append(embeddings[patient]["disease"])
        df = pd.DataFrame({"tsne_1": tsne[:, 0], "tsne_2": tsne[:, 1], "color": c})
        colors = sns.color_palette("hls", len(set(df["color"].values)))

        plt.figure(figsize=(10, 10))

        for index, (i, dff) in enumerate(df.groupby("color")):
            plt.scatter(
                dff["tsne_1"],
                dff["tsne_2"],
                alpha=0.7,
                label=i,
                color=colors[index],
            )
        plt.legend()

        return

    def get_similarities(self, subset=[]):
        if len(subset) == 0:
            subset = self.subset_das28_test
        similarities_mse = {}
        similarities_cos = {}
        for index, p in enumerate(tqdm(subset)):
            if (
                len(self.patient_in_embedding_test[p]["indices"]) > 0
                and len(self.patient_in_embedding_test[p]["indices"]) < 30
            ):
                similarities_mse[p] = compute_similarity(
                    p,
                    subset,
                    self.model_histories_test,
                    self.patient_in_embedding_test,
                    "mse",
                )
                similarities_cos[p] = compute_similarity(
                    p,
                    subset,
                    self.model_histories_test,
                    self.patient_in_embedding_test,
                    "cosine",
                )
        self.similarities_mse = similarities_mse
        self.similarities_cos = similarities_cos

        return similarities_mse, similarities_cos

    # TODO method to highlight one patient, find if some patients move clusters (also potentially in train)
