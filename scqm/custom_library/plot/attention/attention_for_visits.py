from scqm.custom_library.plot.attention.utils import (
    regularise_array,
    get_all_attention_and_ranking,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib


def plot_visit_attention(model, dataset, patient, target_name):
    # patient = '3be81170-6daf-ca49-62a6-fb84ba672b6c'

    _, _, _, _, all_patients_all_attention, _ = get_all_attention_and_ranking(
        model, dataset, [patient], target_name
    )
    values = np.array(
        [
            np.array(
                all_patients_all_attention[patient][visit]["a_visit"].flatten().cpu()
            )
            for visit in all_patients_all_attention[patient].keys()
        ],
        dtype=object,
    )
    arr = values
    val = 0
    lengths = [len(d) for d in arr]
    max_length = max(lengths)
    reg_array = np.zeros(shape=(arr.shape[0], max_length))

    for i in np.arange(arr.shape[0]):
        reg_array[i] = np.append(arr[i], np.zeros(max_length - lengths[i]) + val)
    plt.figure(figsize=(10, 7))
    # plt.grid(True)
    width = 0.99
    n = len(reg_array.T)
    colors = pl.cm.jet(np.linspace(0, 0.7, n))

    for visit in range(len(reg_array.T)):
        plt.bar(
            range(2, len(reg_array.T[0, :]) + 2),
            reg_array.T[visit, :],
            width=width,
            bottom=sum(reg_array.T[visit + 1 :, :]),
            label=f"clinical measures #{visit + 1}",
            color=colors[visit],
            edgecolor="black",
        )
    plt.legend()
    plt.xlabel("Predicted target")
    plt.ylabel("Local attention for clinical measures")
    plt.xticks(range(1, len(reg_array.T[0, :]) + 2))
    plt.xlim([1.5, 9.5])
    plt.ylim([0, 1])
    plt.title("Evolution of attention for local clinical measures")

    return


def aggregate_visit_attention(model, dataset, patients, target_name):
    font = {"family": "normal", "size": 14}

    matplotlib.rc("font", **font)
    _, _, _, _, all_patients_all_attention, _ = get_all_attention_and_ranking(
        model, dataset, patients, target_name
    )
    max_num = 11
    patients_to_keep = [
        patient for patient in patients if len(all_patients_all_attention[patient]) > 0
    ]
    print(f"dropping {len(patients)-len(patients_to_keep)} ")
    max_visit_length = max(
        [
            len(
                all_patients_all_attention[patient][
                    max(all_patients_all_attention[patient].keys())
                ]["a_visit"].flatten()
            )
            if len(
                all_patients_all_attention[patient][
                    max(all_patients_all_attention[patient].keys())
                ]["a_visit"]
            )
            > 0
            else 0
            for patient in patients_to_keep
        ]
    )
    print(max_visit_length)
    visit_histories = {length: [] for length in range(1, max_visit_length + 1)}
    for patient in patients_to_keep:
        values = np.array(
            [
                np.array(
                    all_patients_all_attention[patient][visit]["a_visit"]
                    .flatten()
                    .cpu()
                )
                for visit in all_patients_all_attention[patient].keys()
                if len(all_patients_all_attention[patient][visit]["a_visit"]) > 0
            ],
            dtype=object,
        )
        for value in values:
            visit_histories[len(value)].append(value)
    for length in list(visit_histories.keys()):
        if len(visit_histories[length]) == 0:
            print(length)
            visit_histories.pop(length)
    visit_histories_to_keep = {
        key: elem
        for key, elem in visit_histories.items()
        if len(visit_histories[key]) > 5
    }
    aggregated_histories = np.array(
        [
            sum(visit_histories_to_keep[length]) / len(visit_histories_to_keep[length])
            for length in visit_histories_to_keep.keys()
        ]
    )
    lengths = [len(d) for d in aggregated_histories]
    reg_array = np.zeros(shape=(aggregated_histories.shape[0], max_visit_length))

    for i in np.arange(aggregated_histories.shape[0]):
        reg_array[i] = np.append(
            aggregated_histories[i], np.zeros(max_visit_length - lengths[i])
        )
    # plt.figure(figsize=(10,7))
    fig, ax = plt.subplots(figsize=(6, 5))

    # plt.grid(True)
    width = 0.99

    colors = pl.cm.jet(np.linspace(0, 0.7, max_num))

    for visit in range(max_num):
        plt.bar(
            range(1, max_num + 1),
            reg_array.T[visit, :max_num],
            width=width,
            bottom=sum(reg_array.T[visit + 1 :, :])[:max_num],
            label=f"CM of visit {visit + 1}",
            color=colors[visit],
            edgecolor="black",
        )
    ax.legend(bbox_to_anchor=(1.05, 0.87))
    # plt.legend()
    plt.xlabel("Number of CM available for prediction")
    plt.ylabel("Local attention for CM")
    plt.xticks(range(1, max_num + 2))
    plt.xlim([0.5, max_num + 0.5])
    plt.ylim([0, 1])
    plt.title("Evolution of local attention \n for clinical measures (CM)")
    plt.show()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    return
