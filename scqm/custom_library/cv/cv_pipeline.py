# complete pipeline to load, preprocess, split and save data

import torch
import pickle


from scqm.custom_library.preprocessing.preprocessing import (
    extract_other_features,
    load_dfs_all_data,
    preprocessing,
    extract_adanet_features,
)
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.cv.adaptive_net import CVAdaptivenet

from scqm.custom_library.utils import set_seeds


if __name__ == "__main__":
    model = "other"
    df_dict = load_dfs_all_data()
    if model == "adanet":
        df_dict = preprocessing(df_dict)
        (
            patients_df,
            medications_df,
            visits_df,
            targets_df,
            socioeco_df,
            radai_df,
            haq_df,
            _,
        ) = extract_adanet_features(df_dict, only_meds=True, das28=True)
        df_dict_pro = {
            "a_visit": visits_df,
            "patients": patients_df,
            "med": medications_df,
            "targets": targets_df,
            "socio": socioeco_df,
            "radai": radai_df,
            "haq": haq_df,
        }
    else:
        events_to_keep = [
            "a_visit",
            "med",
            "haq",
            "socioeco",
            "sf_12",
            "euroquol",
            "healthissues",
        ]
        df_dict_pro, events = extract_other_features(
            df_dict,
            transform_meds=True,
            das28=True,
            only_meds=True,
            nan_prop=0.5,
            events_to_keep=events_to_keep,
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_visits = 2
    if model == "adanet":
        dataset = Dataset(
            device,
            df_dict_pro,
            df_dict_pro["patients"]["patient_id"].unique(),
            "das28_increase",
            ["a_visit", "med", "socio", "radai", "haq"],
            min_num_visits,
        )
        dataset.inclusion_criteria()
    else:
        dataset = Dataset(
            device,
            df_dict_pro,
            df_dict_pro["patients"]["patient_id"].unique(),
            "das28_increase",
            events,
            min_num_visits,
        )
        dataset.inclusion_criteria()
    # # keep only patients with more than two visits
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
        ]
    )
    print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    dataset.get_masks(min_time_since_last_event=30, max_time_since_last_event=400)
    if model == "adanet":
        with open("/opt/tmp/saved_dataset_ada_inclusion_400.pickle", "wb") as handle:
            pickle.dump(dataset, handle)
    else:
        with open(
            "/opt/tmp/saved_dataset_inclusion_more_less_bis.pickle", "wb"
        ) as handle:
            pickle.dump(dataset, handle)
    dataset.create_dfs()
    if model == "adanet":
        dataset.transform_to_numeric_adanet()
    else:
        dataset.transform_to_numeric()
    cv = CVAdaptivenet(dataset, k=5)
    if model == "adanet":
        with open("/opt/tmp/saved_cv_ada_inclusion_400.pickle", "wb") as f:
            pickle.dump(cv, f)
    else:
        with open("/opt/tmp/saved_cv_inclusion_more_less_bis.pickle", "wb") as f:
            pickle.dump(cv, f)
