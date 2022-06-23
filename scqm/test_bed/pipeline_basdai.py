# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
import sys
import cProfile
import pstats

sys.path.append("../scqm")
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
import time
import torch
import pandas as pd
import copy
from scqm.test_bed.fake_scqm import get_df_dict
from legacy.scqm.legacy.trainers.multiloss import MultilossTrainer
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from scqm.custom_library.models.other_net_with_double_attention import (
    OthernetWithDoubleAttention,
)
from scqm.custom_library.models.other_net_with_attention import (
    OthernetWithAttention,
)
from scqm.custom_library.models.other_net import Othernet
from scqm.custom_library.models.adaptive_net import Adaptivenet
import sys
import cProfile
import pstats


# setting path


if __name__ == "__main__":
    # create fake data
    df_dict = get_df_dict(num_patients=100)
    real_data = False
    df_dict_processed = copy.deepcopy(df_dict)
    for index, table in df_dict_processed.items():
        date_columns = table.filter(regex=("date")).columns
        df_dict_processed[index][date_columns] = table[date_columns].apply(
            pd.to_datetime
        )
    (
        general_df,
        med_df,
        visits_df,
        basdai_df,
        targets_df_das28,
        targets_df_basdai,
        socioeco_df,
        radai_df,
        haq_df,
    ) = extract_multitask_features(
        df_dict_processed,
        transform_meds=True,
        only_meds=True,
        real_data=False,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": general_df,
        "med": med_df,
        "basdai": basdai_df,
        "targets_basdai": targets_df_basdai,
        "haq": haq_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    dataset = Dataset(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        "None",
        ["a_visit", "med", "haq", "basdai"],
        min_num_targets,
    )
    dataset.drop(
        [id_ for id_, patient in dataset.patients.items() if len(patient.targets) <= 2]
    )
    print(f"Dropping patients with less than 3 targets, keeping {len(dataset)}")
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
    # for key in num_feature_dict:
    #     model_specifics[key] = {
    #         "num_features": num_feature_dict[key],
    #         "size_out": size_out_dict[key],
    #         "size_history": 3,
    #     }
    for key in num_feature_dict:
        model_specifics[key] = {
            "num_features": num_feature_dict[key],
            "size_out": size_out_dict[key],
        }
    model_specifics["size_history"] = 3
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )
    model_specifics["target_name"] = "basdai"
    model = OthernetWithDoubleAttention(model_specifics, device)
    trainer = AdaptivenetTrainer(
        model,
        dataset,
        n_epochs=10,
        batch_size=int(len(dataset) / 2),
        lr=1e-2,
        balance_classes=True,
        use_early_stopping=False,
    )

    # train

    # profiler = cProfile.Profile()
    # profiler.enable()
    start = time.time()
    trainer.train_model(model, partition, debug_patient=False)
    end = time.time()
    print(end - start)
    # test apply function
    for patient in dataset.test_ids:
        (predictions, _, target_values, time_to_targets) = model.apply(
            dataset, dataset.test_ids[0]
        )
    save = False
    if save:
        for name in dataset.df_names:
            getattr(dataset, name).to_csv(
                "scqm/test_bed/dummy_data/" + name + ".csv", index=False
            )

    print("End of script")
