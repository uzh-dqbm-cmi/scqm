import pickle
import re
import os
import pandas as pd


def load_dfs_all_data(subset=None):
    """load all tables"""
    data_path = "/opt/data/01_raw_scqm_all/SCQM_data_tables_all_data_reduced.pickle"
    with open(data_path, "rb") as handle:
        df_dict = pickle.load(handle)

    # rename dfs to match names in previous dataset
    name_mapping = {
        "p": "patients",
        "v": "visits",
        "m": "medications",
        "hi": "healthissues",
        "mny": "modifiednewyorkxrayscore",
        "rau": "ratingenscore",
        "soa": "sonaras",
        "sor": "sonarra",
        "as": "asas",
        "bf": "basfi",
        "bd": "basdai",
        "d": "dlqi",
        "eq": "euroquol",
        "h": "haq",
        "ps": "psada",
        "ra": "radai5",
        "sf": "sf_12",
        "se": "socioeco",
    }
    for key in list(df_dict.keys()):
        df_dict[name_mapping[key]] = df_dict.pop(key)
    for key in df_dict:
        df_dict[key] = df_dict[key].rename(
            columns=lambda x: re.sub("^[^.]*", "", x)[1:]
        )

    for index, table in df_dict.items():
        if df_dict[index].filter(regex=("patient_id")).shape[1] == 1:
            # for consistency
            df_dict[index] = df_dict[index].rename(
                columns=lambda x: re.sub(".*patient_id", "patient_id", x)
            )
        else:
            print(f"table {index} has not ids for patients")
        if "uid_num" or "patient_id" in table.columns:
            pass
        else:
            print("PROBLEM")
    if subset is not None:
        # keep only a subset
        patients_to_keep = subset
        for key in df_dict:
            df_dict[key] = df_dict[key][df_dict[key].patient_id.isin(patients_to_keep)]
    for index, table in df_dict.items():
        print(f"name {index} shape {table.shape}")

    return df_dict
