import numpy as np
import pandas as pd
import copy

from scqm.custom_library.preprocessing.utils import das28_increase
from scqm.custom_library.preprocessing.preprocessing import preprocessing, drop_low_var


def extract_adanet_features(
    df_dict: dict,
    transform_meds: bool = True,
    das28: bool = True,
    only_meds: bool = False,
    joint_df: bool = False,
    real_data: bool = True,
):
    """Extract features for adaptive net from data

    Args:
        df_dict (dict): dictionnary of raw dfs
        transform_meds (bool, optional): whether to transform the medications. Defaults to True.
        das28 (bool, optional): keep only visits with das28. Defaults to True.
        only_meds (bool, optional): keep only patients with available medications. Defaults to False.
        joint_df (bool, optional): return an aggregated dataframe. Defaults to False.
        real_data (bool, optional): real or dummy data. Defaults to True.

    Returns:
        _type_: dataframes with selected features
    """
    general_df = df_dict["patients"][
        [
            "patient_id",
            "date_of_birth",
            "gender",
            "anti_ccp",
            "ra_crit_rheumatoid_factor",
            "date_first_symptoms",
            "date_diagnosis",
        ]
    ]
    med_df = df_dict["medications"][
        [
            "patient_id",
            "med_id",
            "medication_generic_drug",
            "medication_drug_classification",
            "medication_dose",
            "medication_start_date",
            "medication_end_date",
        ]
    ]
    visits_df = df_dict["visits"][
        [
            "patient_id",
            "uid_num",
            "date",
            "weight_kg",
            "das283bsr_score",
            "n_swollen_joints",
            "n_painfull_joints",
            "bsr",
            "n_painfull_joints_28",
            "height_cm",
            "crp",
        ]
    ]
    if real_data:
        socioeco_df = df_dict["socioeco"][["patient_id", "uid_num", "date", "smoker"]]
        radai_df = df_dict["radai5"][
            [
                "patient_id",
                "uid_num",
                "date",
                "pain_level_today_radai",
                "morning_stiffness_duration_radai",
                "activity_of_rheumatic_disease_today_radai",
            ]
        ]
    haq_df = df_dict["haq"][["patient_id", "uid_num", "date", "haq_score"]]
    # keep only some specific medications and change the label of remaining to "other"
    drugs_to_keep = [
        "methotrexate",
        "prednisone",
        "rituximab",
        "adalimumab",
        "sulfasalazine",
        "leflunomide",
        "etanercept",
        "infliximab",
    ]
    med_df.loc[
        ~med_df["medication_generic_drug"].isin(drugs_to_keep),
        "medication_generic_drug",
    ] = "Other"
    # for other_df in [socioeco_df, radai_df, haq_df]:
    #     visits_df = visits_df.merge(other_df, how='outer', on='uid_num')
    if das28:
        # keep only visits with available das28 score
        print(
            f'Dropping {visits_df["das283bsr_score"].isna().sum()} out of {len(visits_df)} visits because das28 is unavailable'
        )
        visits_df = visits_df.dropna(subset=["das283bsr_score"])
        # add column for classification
        visits_df["das28_category"] = [
            0 if value <= 2.6 else 1 for value in visits_df["das283bsr_score"]
        ]
        # keep only subset of patients appearing in visits_id
        patients = visits_df["patient_id"].unique()
        print(
            f"Keeping {len(general_df[general_df.patient_id.isin(patients)])} patients out of {len(general_df.patient_id.unique())}"
        )
        # column saying if das28 at next visit increased (1) or decreased/remained stable 0
        visits_df["das28_increase"] = np.nan
        visits_df = visits_df.groupby("patient_id").apply(das28_increase)
        general_df = general_df[general_df.patient_id.isin(patients)]
        med_df = med_df[med_df.patient_id.isin(patients)]
        haq_df = haq_df[haq_df.patient_id.isin(patients)]
        if real_data:
            socioeco_df = socioeco_df[socioeco_df.patient_id.isin(patients)]
            radai_df = radai_df[radai_df.patient_id.isin(patients)]

    if only_meds:
        patients = med_df["patient_id"].unique()
        print(
            f"keeping only patients with medical info, keeping {len(patients)} out of {len(general_df.patient_id.unique())}"
        )
        general_df = general_df[general_df.patient_id.isin(patients)]
        visits_df = visits_df[visits_df.patient_id.isin(patients)]
        haq_df = haq_df[haq_df.patient_id.isin(patients)]
        if real_data:
            socioeco_df = socioeco_df[socioeco_df.patient_id.isin(patients)]
            radai_df = radai_df[radai_df.patient_id.isin(patients)]

    # sort dfs
    visits_df.sort_values(["patient_id", "date"], inplace=True)
    general_df.sort_values(["patient_id"], inplace=True)
    med_df.sort_values(
        ["patient_id", "medication_start_date", "medication_end_date"], inplace=True
    )
    if das28:
        targets_df = visits_df[
            [
                "patient_id",
                "date",
                "uid_num",
                "das283bsr_score",
                "das28_category",
                "das28_increase",
            ]
        ]
    else:
        targets_df = visits_df[
            [
                "patient_id",
                "date",
                "uid_num",
                "das283bsr_score",
            ]
        ]
    haq_df.sort_values(["patient_id", "date"], inplace=True)
    if real_data:
        socioeco_df.sort_values(["patient_id", "date"], inplace=True)
        radai_df.sort_values(["patient_id", "date"], inplace=True)

    if transform_meds:
        # add new column to med_df indicating for each event if it is a start or end of medication (0 false, 1 true) and replace med_start and med_end
        # by unique column (date). If start date is not available, drop the row. If start and end are available duplicate the row (with different date and is_start dates)
        med_df = med_df.dropna(subset=["medication_start_date"])
        med_df = med_df.rename({"medication_start_date": "date"}, axis=1)
        med_df["is_start"] = 1
        tmp = med_df[med_df.medication_end_date.notna()].copy()
        tmp["date"] = tmp["medication_end_date"]
        tmp["is_start"] = 0
        med_df = pd.concat([med_df, tmp]).drop(columns=["medication_end_date"])
        med_df.sort_values(["patient_id", "date"], inplace=True)

    # create a single df that contains all the info (for the baselines)
    if joint_df:
        visits_df["is_visit"] = 1
        med_df["is_visit"] = 0
        joint_df = pd.concat([visits_df, med_df], ignore_index=True).sort_values(
            by=["patient_id", "date", "uid_num"], axis=0
        )

    else:
        joint_df = []
    if not real_data:
        radai_df = None
        socioeco_df = None
    return (
        general_df,
        med_df,
        visits_df,
        targets_df,
        socioeco_df,
        radai_df,
        haq_df,
        joint_df,
    )


def extract_other_features(
    df_dict: dict,
    transform_meds: bool = True,
    das28: bool = True,
    only_meds: bool = False,
    nan_prop: float = 1,
    events_to_keep=None,
):
    """Extract features from raw dfs

    Args:
        df_dict (dict): dict of raw data
        transform_meds (bool, optional): whether to transform the medications. Defaults to True.
        das28 (bool, optional): keep only visits with available das28. Defaults to True.
        only_meds (bool, optional): keep only patients with available medication. Defaults to False.
        nan_prop (float, optional): threshold to drop columns with too many nans (between 0 and 1, 1 means dropping columns with only missing values). Defaults to 1.
        events_to_keep (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: dataframes with extracted features
    """
    df_dict_processed = copy.deepcopy(df_dict)
    df_dict_processed = preprocessing(df_dict_processed)  #

    if das28:
        # keep only visits with available das28 score
        print(
            f'Dropping {df_dict_processed["visits"]["das283bsr_score"].isna().sum()} out of {len(df_dict_processed["visits"])} visits because das28 is unavailable'
        )
        df_dict_processed["visits"] = df_dict_processed["visits"].dropna(
            subset=["das283bsr_score"]
        )
        # add column for classification
        df_dict_processed["visits"]["das28_category"] = [
            0 if value <= 2.6 else 1
            for value in df_dict_processed["visits"]["das283bsr_score"]
        ]
        # keep only subset of patients appearing in visits_id
        patients = df_dict_processed["visits"]["patient_id"].unique()
        print(
            f'Keeping {len(df_dict_processed["patients"][df_dict_processed["patients"].patient_id.isin(patients)])} patients out of {len(df_dict_processed["patients"].patient_id.unique())}'
        )
        # column saying if das28 at next visit increased (1) or decreased/remained stable 0
        df_dict_processed["visits"]["das28_increase"] = np.nan
        df_dict_processed["visits"] = (
            df_dict_processed["visits"].groupby("patient_id").apply(das28_increase)
        )
        for key in df_dict_processed.keys():
            df_dict_processed[key] = df_dict_processed[key][
                df_dict_processed[key].patient_id.isin(patients)
            ]

    if only_meds:
        patients = df_dict_processed["medications"]["patient_id"].unique()
        print(
            f'keeping only patients with medical info, keeping {len(patients)} out of {len(df_dict_processed["patients"].patient_id.unique())}'
        )
        for key in df_dict_processed.keys():
            df_dict_processed[key] = df_dict_processed[key][
                df_dict_processed[key].patient_id.isin(patients)
            ]

    # sort dfs
    for key in df_dict_processed.keys():
        if key == "patients":
            df_dict_processed[key].sort_values(["patient_id"], inplace=True)
        elif key == "medications":
            df_dict_processed[key].sort_values(
                ["patient_id", "medication_start_date", "medication_end_date"],
                inplace=True,
            )
        else:
            df_dict_processed[key].sort_values(["patient_id", "date"], inplace=True)

    targets_df = df_dict_processed["visits"][
        [
            "patient_id",
            "date",
            "uid_num",
            "das283bsr_score",
            "das28_category",
            "das28_increase",
        ]
    ]
    # keep only subset of medications
    meds_to_keep = [
        "methotrexate",
        "prednisone",
        "adalimumab",
        "rituximab",
        "etanercept",
        "leflunomide",
        "infliximab",
        "sulfasalazine",
        "golimumab",
        "hydroxychloroquine",
        "tocilizumab",
        "certolizumab",
        "abatacept",
        "tofacitinib",
        "secukinumab",
        "baricitinib",
        "prednison_steroid_mr",
        "apremilast",
        "ustekinumab",
    ]
    df_dict_processed["medications"]["medication_generic_drug"] = [
        med if med in meds_to_keep else "Other"
        for med in df_dict_processed["medications"]["medication_generic_drug"]
    ]

    if transform_meds:
        # add new column to med_df indicating for each event if it is a start or end of medication (0 false, 1 true) and replace med_start and med_end
        # by unique column (date). If start date is not available, drop the row. If start and end are available duplicate the row (with different date and is_start dates)
        df_dict_processed["medications"] = df_dict_processed["medications"].dropna(
            subset=["medication_start_date"]
        )
        df_dict_processed["medications"] = df_dict_processed["medications"].rename(
            {"medication_start_date": "date"}, axis=1
        )
        df_dict_processed["medications"]["is_start"] = 1
        tmp = df_dict_processed["medications"][
            df_dict_processed["medications"].medication_end_date.notna()
        ].copy()
        tmp["date"] = tmp["medication_end_date"]
        tmp["is_start"] = 0
        df_dict_processed["medications"] = pd.concat(
            [df_dict_processed["medications"], tmp]
        ).drop(columns=["medication_end_date"])
        df_dict_processed["medications"].sort_values(
            ["patient_id", "date"], inplace=True
        )

    # drop columns with nan proportion equal or more than nan_prop and columns with too low variance
    for index, table in df_dict_processed.items():
        # thresh : require that many non na values to keep the column
        tmp = df_dict_processed[index].dropna(
            axis=1, thresh=int(len(table) * (1 - nan_prop) + 1)
        )
        print(
            f"{index} dropping {df_dict_processed[index].shape[1]-tmp.shape[1]} because more than {nan_prop*100} % missing values"
        )
        df_dict_processed[index] = tmp
        df_dict_processed[index] = drop_low_var(df_dict_processed, index)

    # for coherence
    df_dict_processed["a_visit"] = df_dict_processed.pop("visits")
    df_dict_processed["med"] = df_dict_processed.pop("medications")
    all_events = [
        "a_visit",
        "med",
        "healthissues",
        "modifiednewyorkxrayscore",
        "ratingenscore",
        "sonaras",
        "sonarra",
        "asas",
        "basfi",
        "basdai",
        "dlqi",
        "euroquol",
        "haq",
        "psada",
        "radai5",
        "sf_12",
        "socioeco",
    ]
    if events_to_keep is None:
        events = [
            "a_visit",
            "med",
            "healthissues",
            "ratingenscore",
            "sonarra",
            "euroquol",
            "haq",
            "radai5",
            "sf_12",
            "socioeco",
        ]
    else:
        events = events_to_keep
    for key in set(all_events).difference(events):
        df_dict_processed.pop(key, None)
    df_dict_processed["targets"] = targets_df

    return df_dict_processed, events
