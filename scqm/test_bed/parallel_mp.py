# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
import sys

sys.path.append("../scqm")

import copy

from scqm.test_bed.fake_scqm import get_df_dict
from scqm.custom_library.cv.multitask import CVMultitask
from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
from legacy.scqm.legacy.models.adaptive_net import Adaptivenet
import torch.multiprocessing as mp
import itertools
import random
import numpy as np


import torch
import pandas as pd


from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
import time
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
from scqm.custom_library.models.multitask_net import Multitask


# setting path

if __name__ == "__main__":

    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
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
    min_num_targets = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        "cpu",
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
    cv = CVMultitask(dataset, k=5)
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

    processes = []
    Folds = [0, 1, 2, 3, 4]
    for fold in Folds:

        print(fold)
        device = torch.device(
            "cuda:" + str(fold) if torch.cuda.is_available() else "cpu"
        )
        cv.partition.set_current_fold(fold)
        model_specifics["device"] = device
        model = Multitask(model_specifics, device)
        trainer = MultitaskTrainer(
            model,
            dataset,
            n_epochs=20,
            batch_size={
                "das28": int(len(dataset) / 10),
                "basdai": int(len(dataset) / 10),
            },
            lr=1e-2,
            balance_classes=True,
            use_early_stopping=False,
        )

        p = mp.Process(target=trainer.train_model, args=(model, cv.partition, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        # trainer.train_model(model, cv.partition, debug_patient=False)

    print("End of script")
