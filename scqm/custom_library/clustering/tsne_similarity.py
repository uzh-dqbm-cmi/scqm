import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def get_indices(df, cluster_analysis, tsne):
    patient_id = df.patient_id.iloc[0]
    df = df.sort_values(by="prediction_dates")
    indices = cluster_analysis.patient_in_embedding_test[patient_id]["indices"]
    df["indices"] = indices
    for index in range(tsne.shape[1]):

        df["tsne_" + str(index)] = tsne[indices][:, index]
    return df


def get_distance(position_array, tmp):
    dist = np.zeros((len(position_array[0]), len(position_array[0])))
    for dim in range(len(position_array)):
        dist += (position_array[dim, :] - position_array[dim, :][:, np.newaxis]) ** 2
    dist_df = pd.DataFrame(
        dist, columns=tmp.patient_id + "index" + tmp.indices.astype(str)
    )
    return dist_df


def get_similar_targets(tmp, dist_df):
    similar_targets = []
    for index in tqdm(range(len(tmp))):
        patient_id = tmp.iloc[index].patient_id
        most_similar = [
            elem
            for elem in list(dist_df.iloc[index].sort_values().index)
            if patient_id not in elem
        ][0].split("index")
        most_similar_id = most_similar[0]
        most_similar_index = int(most_similar[1])
        closest_target = tmp[
            (tmp.patient_id == most_similar_id) & (tmp.indices == most_similar_index)
        ].targets.item()
        similar_targets.append(closest_target)
    return similar_targets


def get_random_targets(tmp, seed=0):
    random.seed(seed)
    random_targets = []
    for index in tqdm(range(len(tmp))):
        random_targets.append(random.choice(tmp.targets.values))
    return random_targets


def get_baseline_targets(tmp, tol=1, seed=0):
    random.seed(seed)
    baseline_targets = []
    avrg_len = []
    for index in tqdm(range(len(tmp))):
        patient_id = tmp.iloc[index].patient_id
        prev_value = tmp.iloc[index].prev_values.item()
        close_patients = tmp[tmp.patient_id != patient_id]
        if (
            len(
                close_patients[
                    (prev_value - tol <= close_patients.prev_values)
                    & (close_patients.prev_values <= prev_value + tol)
                ]
            )
            == 0
        ):
            close_patients = close_patients
        else:
            close_patients = close_patients[
                (prev_value - tol <= close_patients.prev_values)
                & (close_patients.prev_values <= prev_value + tol)
            ]

        avrg_len.append(len(close_patients))
        close = random.choice(close_patients.targets.values)
        baseline_targets.append(close)
    print(np.mean(avrg_len))
    return baseline_targets


def similar_patients(df, cluster_analysis, tsne, time_window=[300, 400]):
    tmp = df.copy()
    tmp["days_to_next"] = (tmp.prediction_dates - tmp.prev_dates).apply(
        lambda x: x.days
    )
    tmp = tmp.groupby("patient_id").apply(get_indices, cluster_analysis, tsne)
    df_window = tmp[
        (tmp.days_to_next > time_window[0]) & (tmp.days_to_next < time_window[1])
    ]
    print(f"Keeping {len(df_window)} observations in window out of {len(tmp)}")
    tsne = np.array([df_window["tsne_" + str(index)] for index in range(tsne.shape[1])])
    dist_df = get_distance(tsne, df_window)

    return df_window, dist_df


def get_similarity_performance(
    df, cluster_analysis, width=50, shift=10, tol=1, interval_start=0, interval_end=430
):
    interval_s = list(range(interval_start, interval_end - width, shift))
    interval_e = list(range(interval_start + width, interval_end, shift))
    num_observations = np.empty(len(interval_s))
    similar_targets_mse = np.empty(len(interval_s))
    baseline_targets_mse = np.empty(len(interval_s))
    random_targets_mse = np.empty(len(interval_s))
    similar_targets_mae = np.empty(len(interval_s))
    baseline_targets_mae = np.empty(len(interval_s))
    random_targets_mae = np.empty(len(interval_s))
    tsne = cluster_analysis.tsne_model_test
    for index in tqdm(range(len(interval_s))):
        window_df, dist_df = similar_patients(
            df,
            cluster_analysis,
            tsne,
            time_window=[interval_s[index], interval_e[index]],
        )
        similar_targets = get_similar_targets(window_df, dist_df)
        baseline_targets = get_baseline_targets(window_df, tol)
        random_targets = get_random_targets(window_df)
        num_observations[index] = len(window_df)
        similar_targets_mse[index] = sum(
            (window_df.targets - similar_targets) ** 2
        ) / len(window_df)
        baseline_targets_mse[index] = sum(
            (window_df.targets - baseline_targets) ** 2
        ) / len(window_df)
        random_targets_mse[index] = sum(
            (window_df.targets - random_targets) ** 2
        ) / len(window_df)
        similar_targets_mae[index] = sum(
            abs(window_df.targets - similar_targets)
        ) / len(window_df)
        baseline_targets_mae[index] = sum(
            abs(window_df.targets - baseline_targets)
        ) / len(window_df)
        random_targets_mae[index] = sum(abs(window_df.targets - random_targets)) / len(
            window_df
        )
    return (
        similar_targets_mse,
        baseline_targets_mse,
        random_targets_mse,
        similar_targets_mae,
        baseline_targets_mae,
        random_targets_mae,
    )
