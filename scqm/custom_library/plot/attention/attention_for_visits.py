from scqm.custom_library.plot.attention.utils import regularise_array, get_all_attention
import numpy as np
import matplotlib.pyplot as plt


def plot_visit_attention(model, dataset, patient):
    _, _, _, all_patients_all_attention = get_all_attention(model, dataset, [patient])
    values = np.array(
        [
            np.array(
                all_patients_all_attention[patient][visit]["a_visit"].flatten().cpu()
            )
            for visit in all_patients_all_attention[patient].keys()
        ],
        dtype=object,
    )
    reg_data = regularise_array(values, val=-1)
    # meds = [med_event[1] + ' (' + med_event[0] + ')' for med_event in meds_and_attention[len(meds_and_attention)-1]]
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(reg_data)
    ax.set_xticks(np.arange(reg_data.shape[1]))
    ax.set_xticklabels([i + 1 for i in range(reg_data.shape[1])])
    ax.set_yticks(np.arange(len(reg_data)))
    ax.set_yticklabels(
        ["prediction of visit " + str(index + 2) for index in range(len(reg_data))]
    )
    plt.xlabel("Visit number ")
    plt.title("Attention given to past visits to predict next")
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )
    return


def aggregate_visit_attention(model, dataset, patients, all_patients_all_attention):
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
    aggregated_histories = np.array(
        [
            sum(visit_histories[length]) / len(visit_histories[length])
            for length in visit_histories.keys()
        ]
    )
    reg_data = regularise_array(aggregated_histories, val=-1)
    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(reg_data)
    ax.set_xticks(np.arange(reg_data.shape[1]))
    ax.set_xticklabels([i + 1 for i in range(reg_data.shape[1])])
    ax.set_yticks(np.arange(len(reg_data)))
    ax.set_yticklabels([str(key + 1) for key in visit_histories.keys()])
    plt.ylabel("Prediction number")
    plt.xlabel("Visit history ")
    plt.title("Average attention given to past visits to predict next")
    im.set_cmap("viridis")
    fig.colorbar(
        im,
    )
    # aggregated_histories_norm = np.array([elem * len(elem) for elem in aggregated_histories])

    # reg_data = regularise_array(aggregated_histories_norm, val=-1)
    # fig, ax = plt.subplots(figsize=(20, 20))
    # im = ax.imshow(reg_data)
    # ax.set_xticks(np.arange(reg_data.shape[1]))
    # ax.set_xticklabels([i + 1 for i in range(reg_data.shape[1])])
    # ax.set_yticks(np.arange(len(reg_data)))
    # ax.set_yticklabels([str(key + 1) for key in visit_histories.keys()])
    # plt.ylabel('Prediction number')
    # plt.xlabel('Visit history ')
    # plt.title('Rescaled')
    # im.set_cmap('viridis')
    # fig.colorbar(im,)
    return
    return visit_histories, aggregated_histories
