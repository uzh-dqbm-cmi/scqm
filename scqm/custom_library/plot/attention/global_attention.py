import matplotlib.pyplot as plt
from scqm.custom_library.plot.attention.utils import apply_attention
import numpy as np
import matplotlib.pylab as pl
import matplotlib


def plot_global_attention_one_patient(model, dataset, patient, target_name):

    all_events, all_attention, all_global_attention, events, _, _ = apply_attention(
        model, dataset, patient, target_name
    )
    fig, ax = plt.subplots(figsize=(10, len(all_global_attention) + 2))
    im = ax.imshow([np.array(elem[0].cpu()) for elem in all_global_attention])
    ax.set_yticks(np.arange(len(all_global_attention)))
    ax.set_yticklabels(
        [
            "prediction visit " + str(index)
            for index in range(1, len(all_global_attention) + 1)
        ]
    )
    ax.set_xticks(np.arange(len(dataset.event_names)))
    ax.set_xticklabels(dataset.event_names)
    fig.colorbar(im)
    return all_global_attention


def plot_global_attention(model, dataset, patients, target_name):
    font = {"family": "normal", "size": 14}

    matplotlib.rc("font", **font)
    all_global_attention = []
    global_attention_per_patient = []
    global_attention_save = []
    local_attention_save = []
    for patient in patients:
        _, local_attention, global_attention, _, _, _ = apply_attention(
            model, dataset, patient, target_name
        )
        all_global_attention.extend(global_attention)
        global_attention_save.append(global_attention)
        local_attention_save.append(local_attention)
        global_attention_per_patient.append(
            np.array([np.array(elem.cpu()).squeeze() for elem in global_attention])
        )

    # aggregated plot
    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.imshow(
        sum([np.array(elem[0].cpu()) for elem in all_global_attention]).T
        / len(all_global_attention)
    )
    name_mapping = {
        "a_visit": "Clinical measures",
        "med": "Medications",
        "general": "Demographics",
        "radai": "PROM",
    }
    ax.set_xticks(np.arange(1 + len(dataset.event_names)))
    ax.set_xticklabels(
        ["Demographics"] + [name_mapping[event] for event in dataset.event_names]
    )
    fig.colorbar(im)
    ax.set_yticklabels([])
    ax.get_yaxis().set_ticks([])
    plt.title("Average global attention")
    plt.show()
    lengths = [len(elem) for elem in global_attention_per_patient]
    max_length = min(int(np.percentile(lengths, 90)), 11)
    means = np.zeros(shape=(max_length, len(global_attention_per_patient[0][0])))
    arr = np.array(global_attention_per_patient)
    for num_target in range(max_length):
        norm = 0
        for array in arr:
            if len(array) > num_target:
                means[num_target] += array[num_target]
                norm += 1
        means[num_target] = means[num_target] / norm
    plt.clf()
    fig, ax = plt.subplots(figsize=(6, 6))
    names = ["Demographics"] + [name_mapping[event] for event in dataset.event_names]
    means_reorder = np.array(
        [means[:, 1].T, means[:, 0].T, means[:, 3].T, means[:, 2].T]
    ).T
    width = 1
    n = len(means_reorder[0])
    colors = ["#048A81", "#06D6A0", "#54C6EB", "#8A89C0"]
    for index, event in enumerate(
        ["Clinical measures", "Demographics", "PROM", "Medications"]
    ):
        plt.bar(
            range(1, len(means_reorder[:, 0]) + 1),
            means_reorder[:, index],
            label=event,
            width=width,
            bottom=np.sum(means_reorder[:, :index], axis=1),
            edgecolor="black",
            color=colors[index],
            alpha=0.9,
        )

    ax.legend(bbox_to_anchor=(1.05, 0.28))
    plt.ylabel("Global attention")
    plt.xlabel("Number of clinical measures available for prediction")
    plt.title("Evolution of global attention")
    plt.xticks(range(1, len(means_reorder[:, 0]) + 1))
    # ax.set_xticklabels(['+ str(index) for index in range(1, len(means_reorder[:,0]) + 1)])
    plt.ylim([0, 1])
    plt.xlim([1 - width / 2, len(means_reorder[:, 0]) + 1 - width / 2])
    plt.show()
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)

    return (global_attention_save, local_attention_save)
