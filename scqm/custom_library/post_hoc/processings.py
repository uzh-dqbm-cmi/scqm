import pandas as pd
from tqdm import tqdm
import numpy as np


def prev_value(df, dataset, target_name="das283bsr_score"):
    p = df.patient_id.iloc[0]
    prev_dates = []
    prev_values = []
    if target_name == "das283bsr_score":
        targets_df = dataset[p].targets_das28_df
    else:
        targets_df = dataset[p].targets_asdas_df
    for date in df.prediction_dates:
        prev_dates.append(targets_df[targets_df.date < date].iloc[-1].date)
        prev_values.append(targets_df[targets_df.date < date].iloc[-1][target_name])
    df["prev_dates"] = prev_dates
    df["prev_values"] = prev_values
    return df


def drugs_taken_at_prediction(
    dataset, df_results, subset, target_name="das283bsr_score"
):
    df_all = pd.DataFrame(
        columns=["patient_id", "prediction_dates", "uid_num", "drugs", "drugs_dates"]
    )
    for p in tqdm(subset):
        if target_name == "das283bsr_score":
            df = dataset[p].targets_das28_df.reset_index(drop=True)
        else:
            df = dataset[p].targets_asdas_df.reset_index(drop=True)
        df["drugs"] = np.nan
        df["drugs_dates"] = np.nan
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
                tmp = pd.DataFrame(cropped_timeline, columns=["date", "type", "id"])
                tmp = tmp[tmp.type.str.contains("med")]
                tmp = tmp.drop_duplicates(subset="id", keep=False)
                med = dataset[p].med_df.merge(
                    tmp, left_on=["med_id", "date"], right_on=["id", "date"]
                )
                if len(med) > 0:
                    med_list = med.medication_generic_drug.values
                    date_list = med.date.values
                else:
                    med_list = None
                    date_list = None
                df_all = pd.concat(
                    [
                        df_all,
                        pd.DataFrame(
                            {
                                "patient_id": p,
                                "prediction_dates": date_nth_target,
                                "uid_num": uid_nth_target,
                                "drugs": [med_list],
                                "drugs_dates": [date_list],
                            }
                        ),
                    ],
                    axis=0,
                )
    aggr = pd.merge(
        df_results,
        df_all,
        left_on=["patient_id", "prediction_dates"],
        right_on=["patient_id", "prediction_dates"],
    )
    dates = []
    for index in range(len(aggr)):
        if aggr.iloc[index].drugs_dates is not None:
            dates.append(
                [
                    (aggr.iloc[index].prediction_dates - elem).days
                    for elem in aggr.iloc[index].drugs_dates
                ]
            )
        else:
            dates.append(None)
    aggr["time_since_med"] = dates
    aggr = aggr.groupby("patient_id").apply(
        prev_value, dataset=dataset, target_name=target_name
    )
    aggr["50_improvement"] = aggr.targets <= aggr.prev_values - 0.5 * aggr.prev_values
    aggr["improvement"] = aggr.targets < aggr.prev_values
    aggr["50_improvement_pred"] = (
        aggr.predictions <= aggr.prev_values - 0.5 * aggr.prev_values
    )
    aggr["improvement_pred"] = aggr.predictions < aggr.prev_values
    aggr_exploded = (
        aggr.set_index(
            [
                "patient_id",
                "targets",
                "predictions",
                "prediction_dates",
                "uid_num",
                "prev_dates",
                "prev_values",
                "50_improvement",
                "50_improvement_pred",
                "improvement",
                "improvement_pred",
            ]
        )
        .apply(pd.Series.explode)
        .reset_index()
    )
    return aggr, aggr_exploded
