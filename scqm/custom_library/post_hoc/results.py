from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import timedelta


def find_last_event(timeline, event="a_visit"):
    event_timeline = list(filter(lambda x: x[1] == event, timeline))
    if len(event_timeline) == 0:
        last_event = None
    else:
        last_event = event_timeline[-1]
    return [last_event]


def plot_days_to_last_event(timeline_df, event="last_visit"):
    series = pd.Series(
        timeline_df["prediction_dates"].values
        - pd.Series(
            [elem[0] if elem is not None else np.nan for elem in timeline_df[event]]
        )
    )
    plt.hist([elem.days for elem in series], bins=30)
    return series


def days_to_last_event(dataset, df_results, subset, target_name, no_att=False):
    timeline_df = pd.DataFrame(
        columns=[
            "patient_id",
            "uid_num",
            "timeline",
            "prediction_dates",
            "last_visit",
            "last_radai",
            "last_med_start",
            "last_med_end",
        ]
    )

    for p in tqdm(subset):
        if target_name == "das283bsr_score":
            df = dataset[p].targets_das28_df
            if no_att:
                days = [30, 100, 200, 400, 550, 850]
            else:
                days = [30, 50, 100, 200, 300, 350]
        else:
            df = dataset[p].targets_asdas_df
            days = [30, 100, 200, 300]
        uid_nums = list(df.uid_num)
        for t in range(1, len(df) + 1):
            (
                num_of_each_event,
                cropped_timeline,
                cropped_timeline_mask,
                date_nth_target,
                to_predict,
                uid_nth_target,
            ) = dataset[p].get_cropped_timeline(n=t, target_name=target_name)
            if to_predict and uid_nth_target in uid_nums:
                last_radai = find_last_event(cropped_timeline, "radai")
                last_visit = find_last_event(cropped_timeline, "a_visit")
                last_med_start = find_last_event(cropped_timeline, "med_s")
                last_med_end = find_last_event(cropped_timeline, "med_e")
                timeline_df = pd.concat(
                    [
                        timeline_df,
                        pd.DataFrame(
                            {
                                "patient_id": p,
                                "uid_num": uid_nth_target,
                                "timeline": [cropped_timeline],
                                "prediction_dates": date_nth_target,
                                "last_visit": last_visit,
                                "last_radai": last_radai,
                                "last_med_start": last_med_start,
                                "last_med_end": last_med_end,
                            }
                        ),
                    ],
                    axis=0,
                )
    timeline_df["days_radai"] = plot_days_to_last_event(
        timeline_df, "last_radai"
    ).values
    timeline_df["days_med_start"] = plot_days_to_last_event(
        timeline_df, "last_med_start"
    ).values
    timeline_df["days_med_end"] = plot_days_to_last_event(
        timeline_df, "last_med_end"
    ).values
    timeline_df["days_visit"] = plot_days_to_last_event(
        timeline_df, "last_visit"
    ).values
    aggr = pd.merge(
        df_results,
        timeline_df,
        how="left",
        left_on=["patient_id", "prediction_dates"],
        right_on=["patient_id", "prediction_dates"],
    )
    y = []
    col = "days_visit"
    for i, x in enumerate(days):
        tmp = aggr[aggr[col] < timedelta(days=x)]
        print(len(tmp))
        y.append(((tmp.targets - tmp.predictions) ** 2).sum() / len(tmp))
    # tmp = aggr[aggr[col] > timedelta(days = x)]
    # print(len(tmp))
    # y.append(((tmp.targets -tmp.predictions)**2).sum()/len(tmp))
    print(y)
    plt.figure()
    plt.title("Days since last visit vs MSE")
    plt.plot(days, y, ".--")
    plt.xlabel("Days")
    plt.ylabel("MSE")
    plt.show()
    return aggr
