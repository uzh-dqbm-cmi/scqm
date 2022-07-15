import scipy
from scipy import stats
import torch
import numpy as np
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.plot.attention.utils import (
    apply_attention,
    regularise_array,
    find_drugs,
)
import matplotlib.pyplot as plt

import pandas as pd


def get_heatmaps_for_patient(model, dataset, patient, target_name):
    all_events, all_attention, all_global_attention, events, _, _ = apply_attention(
        model, dataset, patient, target_name
    )
    meds_all = {
        index: find_drugs(
            [(elem[1], elem[2]) for elem in all_events[index] if "med_" in elem[1]],
            patient,
            dataset,
        )
        for index in range(len(all_events))
    }
    meds_and_attention = {
        index: [
            meds_all[index][index_med]
            + (all_attention[index]["med"].flatten()[index_med].item(),)
            for index_med in range(len(meds_all[index]))
        ]
        for index in range(len(all_events))
    }
    values = np.array(
        [
            np.array([med_event[2] for med_event in meds_and_attention[index]])
            for index in range(len(meds_and_attention))
        ],
        dtype=object,
    )
    reg_data = regularise_array(values, val=-1)
    meds = [
        med_event[1] + " (" + med_event[0] + ")"
        for med_event in meds_and_attention[len(meds_and_attention) - 1]
    ]
    fig, ax = plt.subplots(figsize=(len(meds) + 2, len(meds) / 2 + 2))
    im = ax.imshow(reg_data)
    ax.set_xticks(np.arange(len(meds)))
    ax.set_xticklabels(meds)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels([str(index) for index in range(len(values))])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.title("Drug attention for a single patient")
    plt.ylabel("Drug history before prediction")
    plt.xlabel("Drugs")
    for i in range(len(values)):
        for j in range(len(values[i])):
            text = ax.text(
                j, i, np.round(values[i][j], 2), ha="center", va="center", color="red"
            )
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )
    return


def get_lengths_for_drug(meds_and_attention, meds_all, med, patients):
    # med = 'adalimumab'
    histories_with_meds = {}
    lengths = []
    for patient in patients:
        hist_with_meds = [
            (
                meds_and_attention[patient][index],
                len(meds_and_attention[patient][index]),
            )
            for index in meds_and_attention[patient].keys()
            if ("med_s", med) in meds_all[patient][index]
        ]
        if len(hist_with_meds) > 0:
            histories_with_meds[patient] = hist_with_meds
            for hist in hist_with_meds:
                lengths.append(hist[1])
    # histories_with_meds = {patient : [(meds_and_attention[patient][index], len(meds_and_attention[patient][index])) for index in meds_and_attention[patient].keys() if ('med_s', med) in meds_all[patient][index]] for patient in patients}
    return histories_with_meds, lengths


def plot_drug_heatmap(histories_with_meds, lengths, subset_size=15, med="leflunomide"):
    length = stats.mode(np.array(lengths))[0][0]
    final_histories_to_keep = []
    meds = []
    values = []
    for patient in histories_with_meds.keys():
        for elem in histories_with_meds[patient]:
            if elem[1] == length:
                final_histories_to_keep.append((patient, elem[0]))
    if len(final_histories_to_keep) > subset_size:
        final_histories_to_keep = list(
            np.asarray(final_histories_to_keep)[
                np.random.choice(
                    len(final_histories_to_keep), size=subset_size, replace=False
                )
            ]
        )
    patients = [patient_history[0] for patient_history in final_histories_to_keep]
    histories = [patient_history[1] for patient_history in final_histories_to_keep]
    values = np.array(
        [[med_event[2] for med_event in history] for history in histories]
    )
    meds = np.array(
        [[hist[0:2] for hist in history] for history in histories], dtype=object
    )
    # meds = [med_event[1] + ' (' + med_event[0] + ')' for med_event in history]
    fig, ax = plt.subplots(figsize=(len(patients) + 2, len(patients) + 2))
    plt.title(f"{med}: most frequent history length {length}")
    im = ax.imshow(values)
    # ax.set_xticks(np.arange(len(meds)))
    # ax.set_xticklabels(meds)
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_yticklabels(["patient " + str(index) for index in range(len(values))])
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #      rotation_mode="anchor")
    for i in range(len(values)):
        for j in range(len(values[i])):
            if meds[i][j][0] == "med_s" and meds[i][j][1] == med:
                text = ax.text(
                    j,
                    i,
                    np.round(values[i][j], 2),
                    ha="center",
                    va="center",
                    color="red",
                )
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )


def compute_score_per_drug(
    histories_with_meds,
    lengths,
    med="leflunomide",
):
    #     length = stats.mode(np.array(lengths))[0][0]
    # print(f'most frequent history length for drug {med} : {scipy.stats.mode(np.array(lengths))[0][0]}')
    average_scores = []
    for l in range(5, 15):
        final_histories_to_keep = []
        meds = []
        values = []
        for patient in histories_with_meds.keys():
            for elem in histories_with_meds[patient]:
                if elem[1] == l:
                    final_histories_to_keep.append((patient, elem[0]))
        patients = [patient_history[0] for patient_history in final_histories_to_keep]
        histories = [patient_history[1] for patient_history in final_histories_to_keep]
        all_values = np.array(
            [[med_event[2] for med_event in history] for history in histories]
        )
        meds = np.array(
            [[hist[0:2] for hist in history] for history in histories], dtype=object
        )
        indices = {i: [] for i in range(len(meds))}
        for i_s, sequence in enumerate(meds):
            for i_e, elem in enumerate(sequence):
                if elem[0] == "med_s" and elem[1] == med:
                    indices[i_s].append(i_e)
        average_score = {i: [] for i in range(len(meds))}
        for i in range(len(meds)):
            for j in indices[i]:
                average_score[i].append(all_values[i][j])
            average_score[i] = np.mean(average_score[i])
        # print(f'average importance {np.mean(list(average_score.values()))}')
        average_scores.append(np.mean(list(average_score.values())))
    # print(f'Overall average score : {np.mean(average_scores)} +- {np.std(average_scores)}')
    return histories, all_values, meds, average_scores


def plot_aggregated_drug(patients, dataset, meds_all, meds_and_attention, values):
    df = pd.DataFrame(
        index=dataset.med_df["medication_generic_drug"].unique(), columns=range(5, 15)
    )

    for med in dataset.med_df["medication_generic_drug"].unique():
        histories_with_meds, lengths = get_lengths_for_drug(
            meds_and_attention, meds_all, med, patients
        )
        # plt.figure()
        # (_, _, _) = plt.hist(x=lengths, bins='auto', color='#0504aa',
        #                     alpha=0.7, rwidth=0.85)
        # plt.title(med)
        histories, all_values, meds, average_scores = compute_score_per_drug(
            histories_with_meds, lengths, med
        )
        # histories_with_meds, lengths, med

        df.loc[med, :] = average_scores
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.array(df, dtype=np.float32))
    ax.set_xticks(np.arange(len(range(5, 15))))
    ax.set_xticklabels(range(5, 15))
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.index)
    plt.xlabel("length of history")
    plt.ylabel("drug")
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )
    df_rescaled = df * np.array(range(5, 15))

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.array(df_rescaled, dtype=np.float32))
    ax.set_xticks(np.arange(len(range(5, 15))))
    ax.set_xticklabels(range(5, 15))
    ax.set_yticks(np.arange(len(df_rescaled)))
    ax.set_yticklabels(df_rescaled.index)
    plt.title("Rescaled")
    plt.xlabel("length of history")
    plt.ylabel("drug")
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )
    return
