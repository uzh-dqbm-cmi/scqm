from scqm.custom_library.trainers.mlp import MLPTrainer
from scqm.custom_library.models.MLP import MLP
from scqm.custom_library.partition.partition import DataPartition
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.preprocessing.preprocessing import preprocessing
from scqm.custom_library.preprocessing.select_features import extract_adanet_features
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.cv.adaptive_net import CVAdaptivenet
from scqm.custom_library.preprocessing.load_data import load_dfs_all_data
from scqm.custom_library.cv.multitask import CVMultitask
from scqm.custom_library.utils import set_seeds
from scqm.custom_library.results.multiclass_results import MulticlassResults
from scqm.custom_library.results.results import Results
import pandas as pd

import numpy as np
import random
import torch
import pickle
import itertools
from tqdm import tqdm

from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    set_seeds(0)
    print("fit lr")
    target_name = "asdas_score"

    with open(
        "/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint_15_09.pickle", "rb"
    ) as f:
        cv = pickle.load(f)
    dataset = cv.dataset
    partition = cv.partition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if target_name == "das283bsr_score":
        input_size = dataset.joint_das28_df_scaled_tensor_train.shape[1]
        config = {
            "input_size": input_size,
            "output_size": 1,
            "num_hidden": 5,
            "hidden_size": 100,
        }
    elif target_name == "asdas_score":
        input_size = dataset.joint_asdas_df_scaled_tensor_train.shape[1]
        config = {
            "input_size": input_size,
            "output_size": 1,
            "num_hidden": 3,
            "hidden_size": 40,
        }
    for fold in tqdm(range(5)):

        partition.set_current_fold(fold)
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
                    dataset.tensor_indices_mapping_train[patient][
                        "joint_targets_das28_df"
                    ]
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
                    dataset.tensor_indices_mapping_train[patient][
                        "joint_targets_asdas_df"
                    ]
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
        with open(
            "/cluster/work/medinfmk/scqm/tmp/baselines/lr_asdas_bis"
            + str(fold)
            + ".pickle",
            "wb",
        ) as handle:
            pickle.dump(reg, handle)
