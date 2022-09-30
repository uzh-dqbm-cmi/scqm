import sys

sys.path.append("../scqm")

import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
import cProfile
import pstats
from sklearn.cluster import KMeans
from statistics import mean
from scqm.custom_library.models.MLP import MLP
from scqm.custom_library.models.multitask_net import Multitask
from scqm.custom_library.trainers.mlp import MLPTrainer
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
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    seed = 0
    # create fake data
    df_dict = get_df_dict(num_patients=200)
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
        socioeco_df,
        radai_df,
        haq_df,
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
        "targets_das28": targets_df_das28,
        "targets_asdas": targets_df_asdas,
        "haq": haq_df,
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
        ["a_visit", "med", "haq"],
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
    target_name = "das283bsr_score"
    if target_name == "das283bsr_score":
        input_size = dataset.joint_das28_df_scaled_tensor_train.shape[1]
    elif target_name == "asdas_score":
        input_size = dataset.joint_asdas_df_scaled_tensor_train.shape[1]
    config = {"input_size": input_size, "output_size": 1}

    if target_name == "das283bsr_score":
        partitions_test = partition.partitions_test_das28
        partitions_train = partition.partitions_train_das28
        tensor_names = ["joint_das28", "joint_targets_das28"]
        indices_features = np.concatenate(
            [
                dataset.tensor_indices_mapping_train[patient]["joint_das28_df"]
                for patient in partitions_train[partition.current_fold]
                + partition.partitions_train_both[partition.current_fold]
            ]
        )
        indices_targets = np.concatenate(
            [
                dataset.tensor_indices_mapping_train[patient]["joint_targets_das28_df"]
                for patient in partitions_train[partition.current_fold]
                + partition.partitions_train_both[partition.current_fold]
            ]
        )
        train_tensor = dataset.joint_das28_df_scaled_tensor_train[indices_features]
        train_target = dataset.joint_targets_das28_df_scaled_tensor_train[
            indices_targets,
            dataset.joint_targets_das28_df_columns_in_tensor.index("value"),
        ].reshape(len(indices_targets), 1)
    elif target_name == "asdas_score":
        partitions_test = partition.partitions_test_asdas
        partitions_train = partition.partitions_train_asdas
        tensor_names = ["joint_asdas", "joint_targets_asdas"]
        indices_features = np.concatenate(
            [
                dataset.tensor_indices_mapping_train[patient]["joint_asdas_df"]
                for patient in partitions_train[partition.current_fold]
                + partition.partitions_train_both[partition.current_fold]
            ]
        )
        indices_targets = np.concatenate(
            [
                dataset.tensor_indices_mapping_train[patient]["joint_targets_asdas_df"]
                for patient in partitions_train[partition.current_fold]
                + partition.partitions_train_both[partition.current_fold]
            ]
        )
        train_tensor = dataset.joint_asdas_df_scaled_tensor_train[indices_features]
        train_target = dataset.joint_targets_asdas_df_scaled_tensor_train[
            indices_targets,
            dataset.joint_targets_asdas_df_columns_in_tensor.index("value"),
        ].reshape(len(indices_targets), 1)

    reg = LinearRegression(n_jobs=-1).fit(train_tensor, train_target)

    print("End of script")
