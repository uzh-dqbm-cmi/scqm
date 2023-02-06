import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def get_indices(df, cluster_analysis, tsne, embeddings, subset="test"):
    patient_id = df.patient_id.iloc[0]
    df = df.sort_values(by="prediction_dates")
    if subset == "test":
        indices = cluster_analysis.patient_in_embedding_test[patient_id]["indices"]
    elif subset == "train":
        indices = cluster_analysis.patient_in_embedding[patient_id]["indices"]
    df["indices"] = indices
    for index in range(tsne.shape[1]):
        df["tsne_" + str(index)] = tsne[indices][:, index]
    df["embeddings"] = tuple(embeddings[indices].numpy())
    return df


def get_tsne_distance(tsne_test, tsne_train, df_test, df_train):
    dist = np.zeros((len(tsne_test[0]), len(tsne_train[0])))
    for dim in range(len(tsne_test)):
        dist += (tsne_train[dim, :] - tsne_test[dim, :][:, np.newaxis]) ** 2
    dist_df = pd.DataFrame(
        dist, columns=df_train.patient_id + "index" + df_train.indices.astype(str)
    )
    return dist_df


def get_similar_targets(df_test, df_train, dist_df):
    similar_targets = []
    for index in tqdm(range(len(df_test))):
        patient_id = df_test.iloc[index].patient_id
        most_similar = [elem for elem in list(dist_df.iloc[index].sort_values().index)][
            0
        ].split("index")
        most_similar_id = most_similar[0]
        most_similar_index = int(most_similar[1])
        closest_target = df_train[
            (df_train.patient_id == most_similar_id)
            & (df_train.indices == most_similar_index)
        ].targets.item()
        similar_targets.append(closest_target)
    return similar_targets


def get_random_targets(df_test, df_train, num=1, seed=0):
    random.seed(seed)
    random_targets = np.empty((len(df_test), num))
    for index in tqdm(range(len(df_test))):
        random_targets[index, :] = random.sample(list(df_train.targets.values), num)
    return random_targets.mean(axis=1)


def get_baseline_targets(df_test, df_train, tol=1, num=1, seed=0):
    random.seed(seed)
    baseline_targets = np.empty((len(df_test), num))
    avrg_len = []
    for index in tqdm(range(len(df_test))):
        patient_id = df_test.iloc[index].patient_id
        prev_value = df_test.iloc[index].prev_values.item()

        if (
            len(
                df_train[
                    (prev_value - tol <= df_train.prev_values)
                    & (df_train.prev_values <= prev_value + tol)
                ]
            )
            == 0
        ):
            close_patients = df_train
        else:
            close_patients = df_train[
                (prev_value - tol <= df_train.prev_values)
                & (df_train.prev_values <= prev_value + tol)
            ]

        avrg_len.append(len(close_patients))
        close = random.sample(list(close_patients.targets.values), num)
        baseline_targets[index, :] = close
    print(np.mean(avrg_len))
    return baseline_targets.mean(axis=1)


# def similar_patients(df, cluster_analysis, tsne, time_window=[300, 400]):
#     tmp = df.copy()
#     tmp["days_to_next"] = (tmp.prediction_dates - tmp.prev_dates).apply(
#         lambda x: x.days
#     )
#     tmp = tmp.groupby("patient_id").apply(get_indices, cluster_analysis, tsne)
#     df_window = tmp[
#         (tmp.days_to_next > time_window[0]) & (tmp.days_to_next < time_window[1])
#     ]
#     print(f"Keeping {len(df_window)} observations in window out of {len(tmp)}")
#     tsne = np.array([df_window["tsne_" + str(index)] for index in range(tsne.shape[1])])
#     dist_df = get_tsne_distance(tsne, df_window)

#     return df_window, dist_df


def get_similarity_performance(
    df_test,
    df_train,
    cluster_analysis,
    tol=1,
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

    tsne_test = np.array(
        [df_test["tsne_" + str(index)] for index in range(tsne_test.shape[1])]
    )
    tsne_train = np.array(
        [df_train["tsne_" + str(index)] for index in range(tsne_train.shape[1])]
    )

    dist_tsne_df = get_tsne_distance(tsne_test, tsne_train, df_test, df_train)
    similar_targets = get_similar_targets(df_test, df_train, dist_tsne_df)
    baseline_targets = get_baseline_targets(df_test, df_train, tol)
    random_targets = get_random_targets(df_test, df_train)
    similar_targets_mse = sum((df_test.targets - similar_targets) ** 2) / len(df_test)
    baseline_targets_mse = sum((df_test.targets - baseline_targets) ** 2) / len(df_test)
    random_targets_mse = sum((df_test.targets - random_targets) ** 2) / len(df_test)
    similar_targets_mae = sum(abs(df_test.targets - similar_targets)) / len(df_test)
    baseline_targets_mae = sum(abs(df_test.targets - baseline_targets)) / len(df_test)
    random_targets_mae = sum(abs(df_test.targets - random_targets)) / len(df_test)
    return (
        similar_targets_mse,
        baseline_targets_mse,
        random_targets_mse,
        similar_targets_mae,
        baseline_targets_mae,
        random_targets_mae,
    )
