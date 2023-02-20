
import numpy as np
import random
import torch
import pickle
from scqm.custom_library.utils import set_seeds
import xgboost as xgb
import itertools
import pandas as pd

if __name__ == "__main__":
    set_seeds(0)
    np.random.get_state()[1][0]
    target_name = 'das283bsr_score'

    with open(
        "/cluster/work/medinfmk/scqm/tmp/saved_cv_with_joint_10_11.pickle", "rb"
    ) as f:
        cv = pickle.load(f)
    dataset = cv.dataset
    partition = cv.partition
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_rounds = 2000
    max_ = dataset.a_visit_df_scaling_values[1][target_name]
    min_ = dataset.a_visit_df_scaling_values[0][target_name]

    if target_name == 'asdas_score':
        X_all = dataset.joint_asdas_df_scaled_tensor_train
        y_all = dataset.joint_targets_asdas_df_scaled_tensor_train[
            :, dataset.joint_targets_asdas_df_columns_in_tensor.index("value")
        ].reshape(len(dataset.joint_targets_asdas_df_scaled_tensor_train), 1)
        tensor_names = ['joint_asdas_df', 'joint_targets_asdas_df']
        params = {'objective': 'reg:squarederror', 'seed': 0, 'eta': 0.03, 'min_child_weight': 0.0,
                    'max_depth': 3, 'subsample': 0.5, 'colsample_by_tree': 1.0, 'lambda': 3.67}
        partition_train = partition.partitions_train_asdas
        partition_test = partition.partitions_test_asdas

    else:
        X_all = dataset.joint_das28_df_scaled_tensor_train
        y_all = dataset.joint_targets_das28_df_scaled_tensor_train[
            :, dataset.joint_targets_das28_df_columns_in_tensor.index("value")
        ].reshape(len(dataset.joint_targets_das28_df_scaled_tensor_train), 1)
        tensor_names = ['joint_das28_df', 'joint_targets_das28_df']
        params = {'objective': 'reg:squarederror', 'seed': 0, 'eta': 0.03, 'min_child_weight': 1.0,
                    'max_depth': 3, 'subsample': 0.75, 'colsample_by_tree': 0.5, 'lambda': 10}
        partition_train = partition.partitions_train_das28
        partition_test = partition.partitions_test_das28
    
    results = {}
    for fold in range(5):
        partition.set_current_fold(fold)
        train_indices = partition_train[partition.current_fold] + \
            partition.partitions_train_both[partition.current_fold]
        valid_indices = partition_test[partition.current_fold] + \
            partition.partitions_test_both[partition.current_fold]
        indices_train_x = [
            dataset.tensor_indices_mapping_train[patient][tensor_names[0]]
            for patient in train_indices
        ]
        if len(indices_train_x) > 0:
            indices_train_x = np.concatenate(indices_train_x)
        indices_valid_x = [
            dataset.tensor_indices_mapping_train[patient][tensor_names[0]]
            for patient in valid_indices
        ]
        if len(indices_valid_x) > 0:
            indices_valid_x = np.concatenate(indices_valid_x)
        X_train = X_all[indices_train_x]
        X_valid = X_all[indices_valid_x]
        indices_train_y = [
            dataset.tensor_indices_mapping_train[patient][tensor_names[1]]
            for patient in train_indices
        ]
        if len(indices_train_y) > 0:
            indices_train_y = np.concatenate(indices_train_y)
        indices_valid_y = [
            dataset.tensor_indices_mapping_train[patient][tensor_names[1]]
            for patient in valid_indices
        ]
        if len(indices_valid_y) > 0:
            indices_valid_y = np.concatenate(indices_valid_y)
        y_train = y_all[indices_train_y]
        y_valid = y_all[indices_valid_y]
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dvalid = xgb.DMatrix(X_valid, label=y_valid)
        y_valid_rescaled = (y_valid * (max_ - min_) + min_).flatten()
        evallist = [(dtrain, 'train'), (dvalid, 'eval')]

        bst = xgb.train(params, dtrain, num_rounds, evallist, early_stopping_rounds=10)
        preds = bst.predict(xgb.DMatrix(X_valid), iteration_range=(0, bst.best_iteration + 1))
        preds_rescaled = torch.tensor(preds * (max_ - min_) + min_)
        preds_rescaled = torch.tensor(preds * (max_ - min_) + min_)
        mse_ = np.round((torch.sum((preds_rescaled - y_valid_rescaled)**2) / len(y_valid_rescaled)).item(), 5)
        results[fold] = bst


    with open(
        "/cluster/work/medinfmk/scqm/tmp/baselines/xgb_das28_models.pickle",
        "wb",
    ) as handle:
        pickle.dump(results, handle)
