# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
from sklearn.cluster import KMeans
import pstats
import cProfile

import sys

sys.path.append("../scqm")
from scqm.custom_library.models.multitask_net import Multitask

from scqm.custom_library.trainers.multitask_net import MultitaskTrainer

from scqm.test_bed.fake_scqm import get_df_dict
import copy
import pandas as pd
import torch
import time
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.partition.multitask_partition import MultitaskPartition


# setting path

if __name__ == "__main__":
    seed = 0
    # create fake data
    df_dict = get_df_dict(num_patients=500)
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
        mny_df,
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
        "targets_das28": targets_df_das28,
        "targets_basdai": targets_df_basdai,
        "haq": haq_df,
        "basdai": basdai_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        ["das283bsr_score", "basdai_score"],
        ["a_visit", "med", "haq", "basdai"],
        min_num_targets,
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
    partition = MultitaskPartition(dataset, k=3)
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
        }
    model_specifics["size_history"] = 3
    model_specifics["size_embedding"] = max(
        [model_specifics[key]["size_out"] for key in num_feature_dict]
    )
    model = Multitask(model_specifics, device)
    trainer = MultitaskTrainer(
        model,
        dataset,
        n_epochs=10,
        batch_size={"das28": int(len(dataset) / 10), "basdai": int(len(dataset) / 10)},
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
    # kmeans clustering
    subset = dataset.train_ids
    # torch.sum(dataset.masks.available_target_mask == True).item()
    numbers_of_target = [
        torch.sum(
            dataset.masks.available_target_mask[dataset.mapping_for_masks[patient]]
            == True
        ).item()
        for patient in subset
    ]
    histories = torch.empty(size=(sum(numbers_of_target), model.pred_input_size))
    index_in_history = 0
    for index, patient in enumerate(subset):
        _, _, _, hist = model.apply(dataset, patient, return_history=True)
        histories[index_in_history : index_in_history + numbers_of_target[index]] = hist
        index_in_history += numbers_of_target[index]
    k = 3
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(histories)
    # evaluate on test
    subset = dataset.test_ids
    numbers_of_target_test = [
        torch.sum(
            dataset.masks.available_target_mask[dataset.mapping_for_masks[patient]]
            == True
        ).item()
        for patient in subset
    ]
    histories_test = torch.empty(
        size=(sum(numbers_of_target_test), model.pred_input_size)
    )
    index_in_history = 0
    for index, patient in enumerate(subset):
        (predictions, target_values, time_to_targets, hist) = model.apply(
            dataset, patient, return_history=True
        )
        histories_test[
            index_in_history : index_in_history + numbers_of_target_test[index]
        ] = hist
        index_in_history += numbers_of_target_test[index]
        print(
            f"pred {predictions} target values {target_values} time {time_to_targets}"
        )
    # kmeans
    kmeans.predict(histories_test)
    print("End of script")
