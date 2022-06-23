import matplotlib.pyplot as plt
from scqm.custom_library.plot.attention.utils import apply_attention
import numpy as np


def plot_global_attention_one_patient(model, dataset, patient):
    all_events, all_attention, all_global_attention, events = apply_attention(
        model, dataset, patient
    )
    fig, ax = plt.subplots(
        figsize=(len(all_global_attention) + 2, len(all_global_attention) + 2)
    )
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


def plot_global_attention(model, dataset, patients):
    all_global_attention = []

    for patient in patients:
        _, _, global_attention, _ = apply_attention(model, dataset, patient)
        all_global_attention.extend(global_attention)
    if len(patients) < 25:
        fig, ax = plt.subplots(figsize=(5, len(all_global_attention)))
        im = ax.imshow([np.array(elem[0].cpu()) for elem in all_global_attention])
        ax.set_yticks(np.arange(len(all_global_attention)))
        ax.set_xticks(np.arange(len(dataset.event_names)))
        ax.set_xticklabels(dataset.event_names)
        fig.colorbar(im)
    # aggregated plot
    fig, ax = plt.subplots(figsize=(5, 1))
    im = ax.imshow(
        sum([np.array(elem[0].cpu()) for elem in all_global_attention]).T
        / len(global_attention)
    )
    ax.set_xticks(np.arange(len(dataset.event_names)))
    ax.set_xticklabels(dataset.event_names)
    fig.colorbar(im)
    plt.title("Agrr")
    return all_global_attention
