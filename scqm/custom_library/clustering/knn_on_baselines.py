import numpy as np
from sklearn.neighbors import NearestNeighbors

def knn_on_baselines(dataset, partition, device, target_name, k):
    fold = 0
    if target_name == 'asdas_score':
        X_all = dataset.joint_asdas_df_scaled_tensor_train
        y_all = dataset.joint_targets_asdas_df_scaled_tensor_train[
            :, dataset.joint_targets_asdas_df_columns_in_tensor.index("value")
        ].reshape(len(dataset.joint_targets_asdas_df_scaled_tensor_train), 1)
        tensor_names = ['joint_asdas_df', 'joint_targets_asdas_df']
        partition_train = partition.partitions_train_asdas
        partition_test = partition.partitions_test_asdas

    else:
        X_all = dataset.joint_das28_df_scaled_tensor_train
        y_all = dataset.joint_targets_das28_df_scaled_tensor_train[
            :, dataset.joint_targets_das28_df_columns_in_tensor.index("value")
        ].reshape(len(dataset.joint_targets_das28_df_scaled_tensor_train), 1)
        tensor_names = ['joint_das28_df', 'joint_targets_das28_df']
        partition_train = partition.partitions_train_das28
        partition_test = partition.partitions_test_das28
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
    subset = dataset.test_ids
    sub_das28 = [
        p for p in subset if dataset[p].target_name in ["das283bsr_score"]
    ]
    sub_asdas = [
        p for p in subset if dataset[p].target_name in ["asdas_score"]
    ]
    sub_both = [p for p in subset if dataset[p].target_name in ["both"]]
    sub_das28 = sub_das28 + sub_both
    sub_asdas = sub_asdas + sub_both
    if target_name == "das283bsr_score":
        indices_features = np.concatenate(
            [
                dataset.tensor_indices_mapping_test[patient]["joint_das28_df"]
                for patient in sub_das28
            ]
        )
        indices_targets = np.concatenate(
            [
                dataset.tensor_indices_mapping_test[patient][
                    "joint_targets_das28_df"
                ]
                for patient in sub_das28
            ]
        )
        X_test = dataset.joint_das28_df_scaled_tensor_test[
            indices_features
        ]
        y_test = dataset.joint_targets_das28_df_scaled_tensor_test[
            indices_targets,
            dataset.joint_targets_das28_df_columns_in_tensor.index("value"),
        ]
        min_ = dataset.joint_das28_df_scaling_values[0]["das283bsr_score"]
        max_ = dataset.joint_das28_df_scaling_values[1]["das283bsr_score"]
        targ_index = dataset.joint_das28_df_columns_in_tensor.index(
            "das283bsr_score"
        )
    elif target_name == "asdas_score":
        indices_features = np.concatenate(
            [
                dataset.tensor_indices_mapping_test[patient]["joint_asdas_df"]
                for patient in subset
            ]
        )
        indices_targets = np.concatenate(
            [
                dataset.tensor_indices_mapping_test[patient][
                    "joint_targets_asdas_df"
                ]
                for patient in subset
            ]
        )
        X_test = dataset.joint_asdas_df_scaled_tensor_test[
            indices_features
        ]
        y_test = dataset.joint_targets_asdas_df_scaled_tensor_test[
            indices_targets,
            dataset.joint_targets_asdas_df_columns_in_tensor.index("value"),
        ]
        # scaling values
        min_ = dataset.joint_asdas_df_scaling_values[0]["asdas_score"]
        max_ = dataset.joint_asdas_df_scaling_values[1]["asdas_score"]
        targ_index = dataset.joint_asdas_df_columns_in_tensor.index(
            "asdas_score"
        )
    y_test_rescaled = (y_test * (max_ - min_) + min_).flatten()
    knn = NearestNeighbors(n_neighbors=k, n_jobs=-1, metric='l1').fit(X_train)
    dist, indices = knn.kneighbors(X_test)
    preds = X_train[:, targ_index] * (max_ - min_) + min_
    similar_targets = np.empty(shape=(len(X_test), k))
    for index in range(k):
        similar_targets[:, index] = preds[indices[:, index]]
    mse_ = sum((y_test_rescaled - similar_targets.mean(axis=1))**2) / len(y_test_rescaled)
    return mse_
