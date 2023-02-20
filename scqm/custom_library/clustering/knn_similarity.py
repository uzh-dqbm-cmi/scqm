from scqm.custom_library.clustering.tsne_similarity import (
    get_indices,
    get_random_targets,
    get_baseline_targets,
)
import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_knn_similarity_performance(
    df_test, df_train, cluster_analysis, tol=1, k=1, metric="l1"
):
    tsne_test = cluster_analysis.tsne_model_test
    embeddings_test = cluster_analysis.model_histories_test

    tsne_train = cluster_analysis.tsne_model_train
    embeddings_train = cluster_analysis.model_histories

    df_train = df_train[
        ~df_train.patient_id.isin(
            [
                "3c3c9112-a004-394e-2fd3-30cf8e5f7633",
                "8ac7b7d9-7ffd-c0bd-7b48-aea3dcd0f838",
                "f9968820-3320-970c-62b3-0916bf056d4c",
            ]
        )
    ]
    df_test = df_test.groupby("patient_id").apply(
        get_indices, cluster_analysis, tsne_test, embeddings_test, subset="test"
    )
    df_train = df_train.groupby("patient_id").apply(
        get_indices, cluster_analysis, tsne_train, embeddings_train, subset="train"
    )

    X_train = np.array([elem for elem in df_train.embeddings])
    X_test = np.array([elem for elem in df_test.embeddings])

    similar_targets = np.empty(shape=(len(X_test), k))

    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric=metric).fit(X_train)
    dist, indices = knn.kneighbors(X_test)
    for index in range(k):
        similar_targets[:, index] = df_train.iloc[indices[:, index]].targets.values

    similar_targets_mse = sum(
        (df_test.targets - similar_targets.mean(axis=1)) ** 2
    ) / len(df_test)

    # baseline_targets = get_baseline_targets(df_test, df_train, tol, num=k)
    # random_targets = get_random_targets(df_test, df_train, num=k)
    # baseline_targets_mse = sum((df_test.targets - baseline_targets) ** 2) / len(df_test)
    # random_targets_mse = sum((df_test.targets - random_targets) ** 2) / len(df_test)
    baseline_targets_mse, random_targets_mse, baseline_targets_mae, random_targets_mae = [], [], [], []
    similar_targets_mae = sum(
        abs(df_test.targets - similar_targets.mean(axis=1))
    ) / len(df_test)
    # baseline_targets_mae = sum(abs(df_test.targets - baseline_targets)) / len(df_test)
    # random_targets_mae = sum(abs(df_test.targets - random_targets)) / len(df_test)

    return (
        similar_targets_mse,
        baseline_targets_mse,
        random_targets_mse,
        similar_targets_mae,
        baseline_targets_mae,
        random_targets_mae,
    )
