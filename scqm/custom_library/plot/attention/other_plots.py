import pandas as pd
import numpy as np
from scqm.custom_library.plot.attention.utils import apply_attention
import matplotlib.pyplot as plt


def plot_whole_timeline(
    model, dataset, patient_id, target_number=None, target_name="das283bsr_score"
):
    (
        all_events,
        all_attention,
        all_global_attention,
        _,
        target_values,
        predictions,
    ) = apply_attention(model, dataset, patient_id, target_name)
    if target_number is None:
        target_number = len(all_events)
    if target_number > len(all_events):
        raise ValueError("Not enough targets")
    else:
        events = all_events[target_number - 1]
        attention = all_attention[target_number - 1]
        tmp = pd.DataFrame(events, columns=["date", "type", "id"])
        tmp["att"] = np.nan
        types = ["a_visit", "radai", "med"]
        for t in types:
            if len(attention[t]) > 0:
                tmp.loc[tmp.type.str.contains(t), "att"] = [
                    elem.item() for elem in attention[t].cpu().flatten()
                ]
        med = dataset[patient_id].med_df.merge(
            tmp, left_on=["med_id", "date"], right_on=["id", "date"]
        )
        med.replace({"med_s": "start", "med_e": "end"}, inplace=True)
        med["Display"] = (
            med.date.astype(str)
            + ": \n"
            + med.medication_generic_drug
            + " ("
            + med.type
            + ")"
        )
        tmp["Display"] = ""
        tmp.loc[tmp.type.str.contains("vis"), "Display"] = (
            tmp.date.astype(str) + ": \n visit"
        )
        tmp.loc[tmp.type.str.contains("rad"), "Display"] = (
            tmp.date.astype(str) + ": \nradai"
        )
        tmp.loc[tmp.type.str.contains("med"), "Display"] = med["Display"].values
        plt.clf()
        fig, ax = plt.subplots(figsize=(18, 2))
        im = ax.imshow([tmp.att])
        ax.set_yticklabels([])
        ax.get_yaxis().set_ticks([])
        ax.set_xticks(np.arange(len(tmp.att)))
        ax.set_xticklabels(tmp.Display)
        plt.xticks(rotation=90)
        plt.title("Local attention")
        fig.colorbar(
            im,
        )
        plt.show()

    return
