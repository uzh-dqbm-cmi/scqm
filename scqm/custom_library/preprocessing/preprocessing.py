import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import re
from datetime import datetime
import pickle
import copy
from scqm.custom_library.global_vars import REFERENCE_DATE


def preprocess_haq(df):
    mapping = {
        "without_difficulty": 0,
        "with_little_difficulty": 1,
        "with_great_difficulty": 2,
        "impossible": 3,
    }

    return df.replace(to_replace=mapping)


def preprocess_sf12(df):
    mapping = {
        "no_not_limited_at_all": 0,
        "yes_limited_a_little": 1,
        "yes_limited_a_lot": 2,
        "never": 0,
        "seldom": 1,
        "sometimes": 2,
        "quite_often": 3,
        "most_of_the_time": 4,
        "all_the_time": 5,
        "excellent": 0,
        "very_good": 1,
        "good": 2,
        "less_good": 3,
        "bad": 4,
        "not_at_all": 0,
        "a_little_bit": 1,
        "moderately": 2,
        "quite_a_bit": 3,
        "extremely": 4,
    }
    return df.replace(to_replace=mapping)


def preprocess_visits(df):
    df_processed = df.copy()
    df_processed.loc[
        df_processed["date"] > datetime.strptime(REFERENCE_DATE, "%d/%m/%Y"),
        "date",
    ] -= np.timedelta64(1, "Y")
    # preprocess height and weight
    df_processed.loc[df_processed["height_cm"] < 50, "height_cm"] = np.nan
    df_processed.loc[df_processed["weight_kg"] < 30, "weight_kg"] = np.nan
    return df_processed


def preprocess_medications(df):
    df_processed = df.copy()
    df_processed.loc[
        df_processed["medication_dose"].astype(float) > 13000, "medication_dose"
    ] = np.nan
    df_processed.loc[
        df_processed["medication_start_date"] < np.datetime64("1950-01-01"),
        "medication_start_date",
    ] += np.timedelta64(100, "Y")
    df_processed.loc[
        df_processed["stored_start_date"] < np.datetime64("1950-01-01"),
        "stored_start_date",
    ] += np.timedelta64(100, "Y")
    return df_processed


def clean_dates(df_dict: dict) -> dict:
    """specific date cleaning

    Args:
        df_dict (dict): dictionnary of dataframes

    Returns:
        dict: cleaned dict of dataframes
    """
    df_dict_processed = df_dict.copy()
    # some specific outlier preprocessing :
    df_dict_processed["patients"]["date_of_birth"][
        df_dict_processed["patients"]["date_of_birth"] == "1063-05"
    ] = "1963-05"
    df_dict_processed["visits"]["date_of_birth"][
        df_dict_processed["visits"]["date_of_birth"] == "1063-05"
    ] = "1963-05"
    df_dict_processed["visits"]["osteodensitometrie_date"][
        df_dict_processed["visits"]["osteodensitometrie_date"] == "2913-03-01"
    ] = "2013-03-01"
    df_dict_processed["visits"]["osteodensitometrie_date"] = df_dict_processed[
        "visits"
    ]["osteodensitometrie_date"].apply(
        lambda x: "2013-" + x.split("-")[1] + "-" + x.split("-")[2]
        if x is not None and x.split("-")[0] == "2301"
        else x
    )
    df_dict_processed["visits"]["hospital_rehab_stay_due_to_arthritis_date"][
        df_dict_processed["visits"]["hospital_rehab_stay_due_to_arthritis_date"]
        == "3013-11-01"
    ] = "2013-11-01"
    df_dict_processed["patients"]["mnyc_date_positive"] = df_dict_processed["patients"][
        "mnyc_date_positive"
    ].apply(lambda x: x.split("|")[0] if x is not None else x)
    df_dict_processed["visits"]["mnyc_date_positive"] = df_dict_processed["visits"][
        "mnyc_date_positive"
    ].apply(lambda x: x.split("|")[0] if x is not None else x)
    df_dict_processed["medications"]["medication_start_date"][
        df_dict_processed["medications"]["medication_start_date"] == "3021-03-15"
    ] = "2021-03-15"
    df_dict_processed["medications"]["stored_start_date"][
        df_dict_processed["medications"]["stored_start_date"] == "3021-03-15"
    ] = "2021-03-15"
    df_dict_processed["healthissues"]["health_issue_date"][
        df_dict_processed["healthissues"]["health_issue_date"] == "20"
    ] = None
    return df_dict_processed


def cleaning(df_dict: dict) -> dict:
    """some specific cleaning for inconsistencies

    Args:
        df_dict (dict): dictionnary of dataframes

    Returns:
        dict: cleaned dataframes
    """
    for col in ["crp", "reference_area_crp_up_to"]:
        df_dict["visits"][col] = pd.to_numeric(
            df_dict["visits"][col].apply(
                lambda x: x.split("< ")[1]
                if (x is not None and len(x.split("< ")) > 1)
                else x
            )
        )
    df_dict["medications"]["medication_dose"] = df_dict["medications"][
        "medication_dose"
    ].replace("1/32", np.nan)
    df_dict["medications"]["medication_drug"] = df_dict["medications"][
        "medication_drug"
    ].replace({"Methoterxat": "methotrexate"})
    return df_dict


def preprocessing(df_dict: dict, real_data: bool = True) -> dict:
    """preprocessing for dataframes

    Args:
        df_dict (dict): dict of dataframes
        real_data (bool, optional): Real or dummy data. Defaults to True.

    Returns:
        dict: preprocessed dict of dfs
    """
    df_dict_processed = copy.deepcopy(df_dict)

    # some specific outlier preprocessing :
    df_dict_processed = clean_dates(df_dict_processed)
    # other specific preprocessing
    df_dict_processed = cleaning(df_dict_processed)
    # convert string dates to datetime and replace "unknown" by np.nan
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(
            pd.to_datetime, format="%Y/%m/%d"
        )
        df_dict_processed[index] = df_dict_processed[index].replace("unknown", np.nan)

    df_dict_processed["medications"]["last_medication_change"] = pd.to_datetime(
        df_dict_processed["medications"]["last_medication_change"], format="%Y/%m/%d"
    )
    # unify date column name
    for index, elem in df_dict_processed.items():
        name = list(elem.filter(regex="visit_date").columns)
        if len(name) == 1:
            df_dict_processed[index] = elem.rename(columns={name[0]: "date"})
    # some inconsistency in medication and haq dates

    df_dict_processed["haq"] = df_dict_processed["haq"][
        df_dict_processed["haq"]["date"] > np.datetime64("1951-01-01")
    ]
    df_dict_processed["haq"] = preprocess_haq(df_dict_processed["haq"])
    df_dict_processed["sf_12"] = preprocess_sf12(df_dict_processed["sf_12"])
    # errors in visits_df
    df_dict_processed["visits"] = preprocess_visits(df_dict_processed["visits"])
    # errors in med df
    df_dict_processed["medications"] = preprocess_medications(
        df_dict_processed["medications"]
    )
    # manually change date names in other dfs
    df_dict_processed["ratingenscore"] = df_dict_processed["ratingenscore"].rename(
        columns={"imaging_score_scoring_date": "date"}
    )
    # for healthissues, 'health_issue_date' is 90% missing --> use recording time when it is missing
    df_dict_processed["healthissues"].loc[
        df_dict_processed["healthissues"].health_issue_date.isna(), "health_issue_date"
    ] = df_dict_processed["healthissues"][
        df_dict_processed["healthissues"].health_issue_date.isna()
    ].recording_time.values
    df_dict_processed["healthissues"] = df_dict_processed["healthissues"].rename(
        columns={"health_issue_date": "date"}
    )
    df_dict_processed["modifiednewyorkxrayscore"] = df_dict_processed[
        "modifiednewyorkxrayscore"
    ].rename(columns={"imaging_score_scoring_date": "date"})
    # for basdai visit_date is missing in 50% of cases --> replace by recording time
    df_dict_processed["basdai"].loc[
        df_dict_processed["basdai"].date.isna(), "date"
    ] = df_dict_processed["basdai"][
        df_dict_processed["basdai"].date.isna()
    ].recording_time.values
    # for psada visit_date is missing in 50% of cases --> replace by recording time
    df_dict_processed["psada"].loc[
        df_dict_processed["psada"].date.isna(), "date"
    ] = df_dict_processed["psada"][
        df_dict_processed["psada"].date.isna()
    ].recording_time.values
    # for radai5 visit_date is missing in 30% of cases --> replace by recording time
    df_dict_processed["radai5"].loc[
        df_dict_processed["radai5"].date.isna(), "date"
    ] = df_dict_processed["radai5"][
        df_dict_processed["radai5"].date.isna()
    ].recording_time.values
    # drop columns with unique values
    for index, table in df_dict_processed.items():
        cols_to_drop = list(
            df_dict_processed[index]
            .nunique()[df_dict_processed[index].nunique() == 1]
            .index
        )
        print(f"{index} dropping {len(cols_to_drop)} because unique value")
        df_dict_processed[index] = df_dict_processed[index].drop(cols_to_drop, axis=1)
    # socioeco specific preprocessing
    df_dict_processed["socioeco"]["smoker"] = df_dict_processed["socioeco"][
        "smoker"
    ].replace(
        {
            "never_been_smoking": "i_have_never_smoked",
            "smoking_currently": "i_am_currently_smoking",
            "a_former_smoker": "i_am_a_former_smoker_for_more_than_a_year",
        }
    )
    # add id to basdai events
    if "basdai" in df_dict_processed.keys():
        df_dict_processed["basdai"]["event_id"] = [
            "basdai_id" + str(i) for i in range(len(df_dict_processed["basdai"]))
        ]
    # medications specific preprocessing
    # sometimes written bsDMARD or brDMARD for the same thing
    df_dict_processed["medications"][
        "medication_drug_classification"
    ] = df_dict_processed["medications"]["medication_drug_classification"].replace(
        {"bsDMARD": "bDMARD", "brDMARD": "bDMARD"}
    )
    df_dict_processed["medications"]["medication_generic_drug"] = df_dict_processed[
        "medications"
    ]["medication_generic_drug"].replace({"prednison_steroid_mr": "prednisone"})
    # add med_id identifiers to medications df
    df_dict_processed["medications"]["med_id"] = [
        "med_" + str(i) for i in range(len(df_dict_processed["medications"]))
    ]

    df_dict_processed["medications"] = find_drug_categories_and_names(
        df_dict_processed["medications"]
    )
    # drop useless columns
    useless_columns = [
        "recording_time",
        "last_change",
        "workflow_state",
        "institution",
        "type_of_visit",
        "consultant_doctor",
        "authored",
        "when_first_employment_after_finishing_education",
    ]
    for key in df_dict_processed:
        df_dict_processed[key] = df_dict_processed[key].drop(
            columns=useless_columns, errors="ignore"
        )

    return df_dict_processed


def drop_low_var(df_dict: dict, event: str, thresh: float = 0.05) -> pd.DataFrame:
    """Drop columns with too low variance

    Args:
        df_dict (dict): dictionnary of dataframes
        event (str): name of dataframe
        thresh (float, optional): Threshold of variance under which a column should be dropped. Defaults to 0.05.

    Returns:
        pd.DataFrame: df_dict[event] with the dropped columns
    """
    to_consider = [
        col
        for col in df_dict[event].columns
        if df_dict[event][col].dtype not in ["<M8[ns]", "O"]
    ]
    normalized = (df_dict[event][to_consider] - df_dict[event][to_consider].min()) / (
        df_dict[event][to_consider].max() - df_dict[event][to_consider].min()
    )
    std = normalized.std()
    to_drop = [
        col for col in normalized.columns if (std[col] <= thresh) or np.isnan(std[col])
    ]
    print(f"dropping {len(to_drop)} columns because of too low variance")
    return df_dict[event].drop(columns=to_drop)


def find_drug_categories_and_names(df: pd.DataFrame) -> pd.DataFrame:
    """Find missing drug categories and names in df medications
    Args:
        df (pd.DataFrame): medication df

    Returns:
        pd.DataFrame: completed df
    """

    # TODO complete for new data
    # Reassmbly of the medication
    # Create a drug to category dictionary to impute any missing category values
    drugs_per_cat = df.groupby(by="medication_drug_classification")[
        "medication_drug"
    ].apply(list)
    drugs_per_generic = df.groupby(by="medication_generic_drug")[
        "medication_drug"
    ].apply(list)
    drug_to_category = dict()
    drug_to_generic = dict()

    for i in range(len(drugs_per_cat)):
        list_of_subcat_drug = list(set(drugs_per_cat[i]))
        for j in list_of_subcat_drug:
            drug_to_category[j] = drugs_per_cat.index[i]

    for i in range(len(drugs_per_generic)):
        list_of_drugs_in_genericdrug = list(set(drugs_per_generic[i]))
        for j in list_of_drugs_in_genericdrug:
            drug_to_generic[j] = drugs_per_generic.index[i]

    # Manually add the missing drugs:
    drug_to_category["spiricort"] = "steroid"
    drug_to_category["Imurek (Azathioprin)"] = "csDMARD"
    drug_to_generic["spiricort"] = "spiricort"
    drug_to_generic["Imurek (Azathioprin)"] = "azathioprine"

    # Display the medications without category
    not_cat_drugs = list(
        set(df["medication_drug"].to_list()) - set(list(drug_to_category.keys()))
    )
    print(not_cat_drugs)

    # Display the medications without generic label
    no_generic_label = list(
        set(df["medication_drug"].to_list()) - set(list(drug_to_generic.keys()))
    )
    print(no_generic_label)

    # Impute all missing medication category based on the created category dictionary
    for i in df.index:
        if pd.isnull(df["medication_drug_classification"][i]):
            if df["medication_drug"][i] not in not_cat_drugs:
                df["medication_drug_classification"][i] = drug_to_category[
                    df["medication_drug"][i]
                ]

    print(df["medication_drug_classification"].unique())
    # Impute all missing generic medication label based on the created generic label dictionary

    for i in df.index:
        if pd.isnull(df["medication_generic_drug"][i]):
            if df["medication_drug"][i] not in no_generic_label:
                df["medication_generic_drug"][i] = drug_to_generic[
                    df["medication_drug"][i]
                ]

    return df
