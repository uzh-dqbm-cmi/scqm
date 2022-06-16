from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.preprocessing.preprocessing import preprocessing
from scqm.custom_library.preprocessing.select_features import (
    extract_adanet_features,
    extract_other_features,
)
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.cv.adaptive_net import CVAdaptivenet
from scqm.custom_library.utils import set_seeds

from scqm.custom_library.models.other_net_multiloss import OthernetMultiloss
from scqm.custom_library.trainers.multiloss import MultilossTrainer

import torch
import pickle

import sys

if __name__ == "__main__":
    model = str(sys.argv[1])
    set_seeds(0)
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
        events = ["a_visit", "med", "socio", "radai", "haq"]
    else:
        df_dict_pro, events = extract_other_features(
            df_dict, transform_meds=True, das28=True, only_meds=True, nan_prop=0.2
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_visits = 2
    dataset = Dataset(
        device,
        df_dict_pro,
        df_dict_pro["patients"]["patient_id"].unique(),
        "das28_increase",
        events,
        min_num_visits,
    )
    # # keep only patients with more than two visits
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
        ]
    )
    print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    dataset.get_masks(min_time_since_last_event=15, max_time_since_last_event=450)
    if model == "adanet":
        path = "/opt/tmp/saved_dataset.pickle"
    else:
        path = "/opt/data/processed/saved_dataset_more_features.pickle"
    with open(path, "wb") as handle:
        pickle.dump(dataset, handle)
    # create cvs
    dataset.create_dfs()
    if model == "adanet":
        dataset.transform_to_numeric_adanet()
        cv = CVAdaptivenet(dataset, k=5)
        with open("/opt/tmp/saved_cv_ada.pickle", "wb") as f:
            pickle.dump(cv, f)
    else:
        dataset.transform_to_numeric()
        cv = CVAdaptivenet(dataset, k=5)
        with open("/opt/data/processed/saved_cv.pickle", "wb") as f:
            pickle.dump(cv, f)

    dataset = cv.dataset
    partition = cv.partition
    print("start")
    fold = int(0)
    partition.set_current_fold(fold)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"fold {fold}")
    num_feature_dict = {
        event: getattr(dataset, event + "_df_scaled_tensor_train").shape[1]
        for event in dataset.event_names
    }
    size_out_dict = {
        event: int(num_feature_dict[event] / 10) + 1 for event in dataset.event_names
    }
    num_feature_dict["patients"] = getattr(
        dataset, "patients" + "_df_scaled_tensor_train"
    ).shape[1]
    size_out_dict["patients"] = int(num_feature_dict["patients"] / 10) + 1
    # size_history_dict = {event: num_feature_dict[event]*2 for event in dataset.event_names}
    model_specifics = {
        "num_layers_enc": 2,
        "hidden_enc": 100,
        "size_history": 10,
        "num_layers": 1,
        "num_layers_pred": 2,
        "hidden_pred": 100,
        "event_names": dataset.event_names,
        "num_general_features": dataset.patients_df_scaled_tensor_train.shape[1],
        "dropout": 0.0,
        "batch_first": True,
        "device": device,
    }
    for key in num_feature_dict:
        model_specifics[key] = {
            "num_features": num_feature_dict[key],
            "size_out": size_out_dict[key],
        }
    model_specifics["size_history"] = 30
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )

    model = OthernetMultiloss(model_specifics, device)

    batch_size = int(len(dataset) / 15)
    trainer = MultilossTrainer(
        model,
        dataset,
        n_epochs=40,
        batch_size=batch_size,
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
    )
    trainer.train_model(model, partition, debug_patient=False)

    delattr(trainer, "dataset")
    with open("/opt/tmp/trainer_multiloss.pickle", "wb") as handle:
        pickle.dump(trainer, handle)
