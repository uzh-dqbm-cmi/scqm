from datetime import timedelta
import pandas as pd
import numpy as np


def ml_prepro(X_train, y_train, X_valid, y_valid, X_test, y_test):
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = seed)
    feat_min_ = np.nanmin(np.concatenate((X_train, X_valid)), axis=0)
    feat_max_ = np.nanmax(np.concatenate((X_train, X_valid)), axis=0)
    targ_min_ = np.nanmin(np.concatenate((y_train, y_valid)), axis=0)
    targ_max_ = np.nanmax(np.concatenate((y_train, y_valid)), axis=0)
    X_train_scaled = (X_train - feat_min_) / (feat_max_ - feat_min_)
    y_train_scaled = (y_train - targ_min_) / (targ_max_ - targ_min_)
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=-1)
    X_valid_scaled = (X_valid - feat_min_) / (feat_max_ - feat_min_)
    y_valid_scaled = (y_valid - targ_min_) / (targ_max_ - targ_min_)
    X_valid_scaled = np.nan_to_num(X_valid_scaled, nan=-1)
    X_test_scaled = (X_test - feat_min_) / (feat_max_ - feat_min_)
    y_test_scaled = (y_test - targ_min_) / (targ_max_ - targ_min_)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=-1)
    return (
        X_train_scaled,
        X_valid_scaled,
        X_test_scaled,
        y_train_scaled,
        y_valid_scaled,
        y_test_scaled,
        feat_min_,
        feat_max_,
        targ_min_,
        targ_max_,
    )


def dfs_as_numeric(features, targets, patients_train, patients_valid, patients_test):
    df = features.astype(
        {
            "hist_bDMARD": int,
            "hist_csDMARD": int,
            "hist_steroid": int,
            "hist_tsDMARD": int,
            "curr_bDMARD": int,
            "curr_csDMARD": int,
            "curr_steroid": int,
            "curr_tsDMARD": int,
            "hist_total": int,
            "curr_total": int,
        }
    )
    df = pd.get_dummies(
        df,
        columns=[
            "anti_ccp",
            "ra_crit_rheumatoid_factor",
            "smoker",
            "morning_stiffness_duration_radai",
        ],
        drop_first=True,
        dummy_na=True,
    )
    df = pd.get_dummies(df, columns=["gender"], drop_first=True)
    REFERENCE_DATE = "01/05/2022"
    for date_col in [
        "date_of_birth",
        "date_first_symptoms",
        "date_diagnosis",
        "time_to_pred",
    ]:
        df[date_col] = (
            pd.to_datetime(REFERENCE_DATE, format="%d/%m/%Y") - df[date_col]
        ).dt.days
    X_train = (
        df[df.patient_id.isin(patients_train)].drop(columns=["patient_id"]).to_numpy()
    )
    X_valid = (
        df[df.patient_id.isin(patients_valid)].drop(columns=["patient_id"]).to_numpy()
    )
    X_test = (
        df[df.patient_id.isin(patients_test)].drop(columns=["patient_id"]).to_numpy()
    )
    y_train = (
        targets[targets.patient_id.isin(patients_train)]
        .drop(columns=["patient_id", "uid_num", "date"])
        .to_numpy()
    )
    y_valid = (
        targets[targets.patient_id.isin(patients_valid)]
        .drop(columns=["patient_id", "uid_num", "date"])
        .to_numpy()
    )
    y_test = (
        targets[targets.patient_id.isin(patients_test)]
        .drop(columns=["patient_id", "uid_num", "date"])
        .to_numpy()
    )

    return (
        df.drop(columns=["patient_id"]),
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
    )


def prepare_features(dataset, patients, target_name):
    medications = dataset.med_df.copy()
    medications.drop(
        columns=["medication_generic_drug", "medication_dose"], inplace=True
    )
    medications.rename(
        columns={"medication_drug_classification": "drugs"}, inplace=True
    )
    medications = pd.get_dummies(medications, columns=["drugs"])
    # meds_final_df = pd.DataFrame(columns = ['patient_id', 'hist_bDMARD', 'hist_csDMARD', 'hist_steroid', 'hist_tsDMARD', 'curr_bDMARD', 'curr_csDMARD', 'curr_steroid', 'curr_tsDMARD'])
    visits = dataset.a_visit_df.copy()
    radai = dataset.radai_df.copy()
    radai.drop(columns=["uid_num"], inplace=True)
    visits["bmi"] = visits["weight_kg"] / (visits["height_cm"] / 100) ** 2
    visits.drop(columns=["weight_kg", "height_cm", "uid_num"], inplace=True)
    patients_df = dataset.patients_df.copy()
    min_time_since_last_event = 15
    meds_final_df = pd.DataFrame(
        columns=[
            "patient_id",
            "hist_bDMARD",
            "hist_csDMARD",
            "hist_steroid",
            "hist_tsDMARD",
            "curr_bDMARD",
            "curr_csDMARD",
            "curr_steroid",
            "curr_tsDMARD",
            "hist_total",
            "curr_total",
        ]
    )
    targets_df = pd.DataFrame(columns=["patient_id", "uid_num", "date", "value"])
    visits_final_df = pd.DataFrame(columns=visits.columns)
    radai_final_df = pd.DataFrame(columns=radai.columns)
    patients_final_df = pd.DataFrame(columns=patients_df.columns)
    for patient in patients:
        for index in range(len(dataset[patient].targets_to_predict[target_name])):
            data = dataset[patient].targets_to_predict[target_name][index].data
            date = data.date.item()
            value = data[target_name].item()
            uid_num = data.uid_num.item()
            # medications
            meds = medications[medications.patient_id == patient]
            meds = meds[(date - meds.date) > timedelta(days=min_time_since_last_event)]
            targets_df = pd.concat(
                [
                    targets_df,
                    pd.DataFrame(
                        {
                            "patient_id": patient,
                            "uid_num": uid_num,
                            "date": date,
                            "value": value,
                        },
                        index=[0],
                    ),
                ]
            )
            if len(meds) > 0:
                drugs_taking = meds.drop_duplicates(subset="med_id", keep=False)
                drugs_history = meds[meds.duplicated(subset="med_id", keep="last")]
                if len(drugs_history) == 0:
                    hist_sum = pd.Series(
                        {
                            "drugs_bDMARD": 0,
                            "drugs_csDMARD": 0,
                            "drugs_steroid": 0,
                            "drugs_tsDMARD": 0,
                        }
                    )
                    hist_total = 0
                else:
                    hist_sum = drugs_history.sum()
                    hist_total = (
                        hist_sum["drugs_bDMARD"]
                        + hist_sum["drugs_csDMARD"]
                        + hist_sum["drugs_steroid"]
                        + hist_sum["drugs_tsDMARD"]
                    )
                if len(drugs_taking) == 0:
                    curr_sum = pd.Series(
                        {
                            "drugs_bDMARD": 0,
                            "drugs_csDMARD": 0,
                            "drugs_steroid": 0,
                            "drugs_tsDMARD": 0,
                        }
                    )
                    curr_total = 0
                else:
                    curr_sum = drugs_taking.sum()
                    curr_total = (
                        curr_sum["drugs_bDMARD"]
                        + curr_sum["drugs_csDMARD"]
                        + curr_sum["drugs_steroid"]
                        + curr_sum["drugs_tsDMARD"]
                    )
                tmp = pd.DataFrame(
                    {
                        "patient_id": patient,
                        "hist_bDMARD": hist_sum["drugs_bDMARD"],
                        "hist_csDMARD": hist_sum["drugs_csDMARD"],
                        "hist_steroid": hist_sum["drugs_steroid"],
                        "hist_tsDMARD": hist_sum["drugs_tsDMARD"],
                        "curr_bDMARD": curr_sum["drugs_bDMARD"],
                        "curr_csDMARD": curr_sum["drugs_csDMARD"],
                        "curr_steroid": curr_sum["drugs_steroid"],
                        "curr_tsDMARD": curr_sum["drugs_tsDMARD"],
                        "hist_total": hist_total,
                        "curr_total": curr_total,
                    },
                    index=[0],
                )
            else:
                tmp = pd.DataFrame(
                    {
                        "patient_id": patient,
                        "hist_bDMARD": 0,
                        "hist_csDMARD": 0,
                        "hist_steroid": 0,
                        "hist_tsDMARD": 0,
                        "curr_bDMARD": 0,
                        "curr_csDMARD": 0,
                        "curr_steroid": 0,
                        "curr_tsDMARD": 0,
                        "hist_total": 0,
                        "curr_total": 0,
                    },
                    index=[0],
                )

            meds_final_df = pd.concat([meds_final_df, tmp])
            # other dfs
            vis = visits[visits.patient_id == patient]
            vis = (
                vis[(date - vis.date) > timedelta(days=min_time_since_last_event)]
                .ffill(axis=0)
                .iloc[-1]
            )
            visits_final_df = pd.concat([visits_final_df, pd.DataFrame([vis])])
            rad = radai[radai.patient_id == patient]
            pat = patients_df[patients_df.patient_id == patient]
            if (
                len(rad) == 0
                or len(
                    rad[(date - rad.date) > timedelta(days=min_time_since_last_event)]
                )
                == 0
            ):
                rad = pd.DataFrame(
                    {
                        "patient_id": patient,
                        "date": np.nan,
                        "pain_level_today_radai": np.nan,
                        "morning_stiffness_duration_radai": np.nan,
                        "activity_of_rheumatic_disease_today_radai": np.nan,
                    },
                    index=[0],
                )

            else:
                rad = pd.DataFrame(
                    [
                        rad[
                            (date - rad.date)
                            > timedelta(days=min_time_since_last_event)
                        ]
                        .ffill(axis=0)
                        .iloc[-1]
                    ]
                )
            radai_final_df = pd.concat([radai_final_df, rad])
            patients_final_df = pd.concat([patients_final_df, pat])
    features = pd.concat(
        [
            meds_final_df.reset_index(drop=True),
            visits_final_df.reset_index(drop=True).drop(columns=["patient_id", "date"]),
            radai_final_df.reset_index(drop=True).drop(columns=["patient_id", "date"]),
            patients_final_df.reset_index(drop=True).drop(columns="patient_id"),
        ],
        axis=1,
    )
    features["time_to_pred"] = targets_df.reset_index(drop=True).date

    return features, targets_df
