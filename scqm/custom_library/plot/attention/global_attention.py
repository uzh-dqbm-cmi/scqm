import matplotlib.pyplot as plt
from scqm.custom_library.plot.attention.utils import apply_attention
import numpy as np


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
    all_global_attention = []
    global_attention_per_patient = []
    for patient in patients:
        _, _, global_attention, _, _, _ = apply_attention(
            model, dataset, patient, target_name
        )
        all_global_attention.extend(global_attention)
        global_attention_per_patient.append(
            np.array([np.array(elem.cpu()).squeeze() for elem in global_attention])
        )

    # aggregated plot
    fig, ax = plt.subplots(figsize=(8, 2))
    im = ax.imshow(
        sum([np.array(elem[0].cpu()) for elem in all_global_attention]).T
        / len(all_global_attention)
    )
    ax.set_xticks(np.arange(1 + len(dataset.event_names)))
    ax.set_xticklabels(["general"] + dataset.event_names)
    fig.colorbar(im)
    plt.title("Global event attention")
    lengths = [len(elem) for elem in global_attention_per_patient]
    max_length = int(np.percentile(lengths, 90))
    means = np.zeros(shape=(max_length, len(global_attention_per_patient[0][0])))
    arr = np.array(global_attention_per_patient)
    for num_target in range(max_length):
        norm = 0
        for array in arr:
            if len(array) > num_target:
                means[num_target] += array[num_target]
                norm += 1
        means[num_target] = means[num_target] / norm

    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(means)
    ax.set_xticks(np.arange(1 + len(dataset.event_names)))
    ax.set_xticklabels(["general"] + dataset.event_names)

    return all_global_attention
