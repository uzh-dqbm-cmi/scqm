# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
import sys

sys.path.append("../scqm")

from scqm.custom_library.models import Adaptivenet, Othernet2
from scqm.custom_library.training import AdaptivenetTrainer
from scqm.test_bed.fake_scqm import get_df_dict

import copy
import pandas as pd
import torch

# setting path

from scqm.custom_library.preprocessing import extract_adanet_features
from scqm.custom_library.data_objects import Dataset
from scqm.custom_library.partition import DataPartition

if __name__ == "__main__":
    # create fake data
    df_dict = get_df_dict()
    real_data = False
    df_dict_processed = copy.deepcopy(df_dict)
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(
            pd.to_datetime
        )
    (
        patients_df,
        medications_df,
        visits_df,
        targets_df,
        _,
        _,
        haq_df,
        _,
    ) = extract_adanet_features(
        df_dict_processed,
        transform_meds=True,
        das28=True,
        only_meds=True,
        joint_df=False,
        real_data=False,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": patients_df,
        "med": medications_df,
        "targets": targets_df,
        "haq": haq_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_visits = 2
    # instantiate dataset
    dataset = Dataset(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        "das28_increase",
        ["a_visit", "med", "haq"],
        min_num_visits,
    )
    dataset.drop(
        [
            id_
            for id_, patient in dataset.patients.items()
            if len(patient.visit_ids) <= 2
        ]
    )
    print(f"Dropping patients with less than 3 visits, keeping {len(dataset)}")
    dataset.get_masks()
    dataset.create_dfs()
    # prepare for training
    dataset.transform_to_numeric_adanet(real_data)
    partition = DataPartition(dataset, k=3)
    fold = int(0)
    partition.set_current_fold(fold)
    num_feature_dict = {
        event: getattr(dataset, event + "_df_scaled_tensor_train").shape[1]
        for event in dataset.event_names
    }
    size_out_dict = {event: 2 for event in dataset.event_names}
    num_feature_dict["patients"] = getattr(
        dataset, "patients" + "_df_scaled_tensor_train"
    ).shape[1]
    size_out_dict["patients"] = 2
    model_specifics = {
        "task": "regression",
        "num_targets": 1,
        "size_embedding": 2,
        "num_layers_enc": 2,
        "hidden_enc": 10,
        "size_history": 2,
        "num_layers": 1,
        "num_layers_pred": 2,
        "hidden_pred": 10,
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
            "size_history": 3,
        }
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )
    model = Othernet2(model_specifics, device)
    trainer = AdaptivenetTrainer(
        model,
        dataset,
        n_epochs=2,
        batch_size=int(len(dataset) / 2),
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
    )
    # train
    trainer.train_model(model, partition, debug_patient=True)

    print("End of script")
