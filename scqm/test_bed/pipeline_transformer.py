# Execute complete timeline with fake data (generate + create dataset object + instantiate model + training)
import sys

sys.path.append("../scqm")

from tqdm import tqdm
import cProfile
import pstats
from sklearn.cluster import KMeans
from statistics import mean
from legacy.scqm.legacy.models.transformer import TransformerModel
from scqm.custom_library.trainers.multitask_net import MultitaskTrainer
from scqm.test_bed.fake_scqm import get_df_dict
import copy
import pandas as pd
import torch
import time
from scqm.custom_library.preprocessing.select_features import extract_multitask_features
from scqm.custom_library.data_objects.dataset_multitask import DatasetMultitask
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.clustering.similarity import compute_similarity
from scqm.custom_library.clustering.utils import (
    get_features,
    get_histories_and_features,
)
import numpy as np


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
        targets_df_das28,
        targets_df_asdas,
        radai_df,
        joint_df,
    ) = extract_multitask_features(
        df_dict_processed,
        transform_meds=True,
        only_meds=True,
        real_data=False,
        joint_df=True,
    )
    df_dict_fake = {
        "a_visit": visits_df,
        "patients": general_df,
        "med": med_df,
        "radai": radai_df,
        "targets_das28": targets_df_das28,
        "targets_asdas": targets_df_asdas,
        "joint": joint_df,
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    min_num_targets = 2
    # instantiate dataset
    dataset = DatasetMultitask(
        device,
        df_dict_fake,
        df_dict_fake["patients"]["patient_id"].unique(),
        ["das283bsr_score", "asdas_score"],
        ["a_visit", "med", "radai"],
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
    dataset.post_process_joint_df()
    dataset.create_dfs()
    # get targets for baseline
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
        "num_layers": 1,
        "num_layers_pred": 2,
        "hidden_pred": 10,
        "event_names": dataset.event_names,
        "num_general_features": dataset.patients_df_scaled_tensor_train.shape[1],
        "dropout": 0.0,
        "batch_first": True,
        "device": device,
        "n_heads": 4,
        "dim_val": 64,
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
    model = TransformerModel(model_specifics, device)
    trainer = MultitaskTrainer(
        model,
        dataset,
        n_epochs=10,
        batch_size={"das28": int(len(dataset) / 10), "asdas": int(len(dataset) / 10)},
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
    for p in dataset.test_ids:
        if dataset[p].target_name in ["both", "das283bsr_score"]:
            model.apply(dataset, p, "das283bsr_score", return_history=False)
        if dataset[p].target_name in ["both", "asdas_score"]:
            model.apply(dataset, p, "asdas_score", return_history=False)
