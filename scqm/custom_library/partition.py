import math
import numpy as np

class DataPartition:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.split()

    def split(self):
        #TODO implement stratifier (on the targets)
        # split data into train and test (no valid)
        self.dataset.split_data(prop_valid=0.0, prop_test=0.2)
        self.dataset.scale_and_tensor()
        # get partition of size k of train set
        self.fold_size = math.ceil(len(self.dataset.train_ids) / self.k)
        self.permuted_ids = np.random.permutation(self.dataset.train_ids)
        self.partitions_test = {test_fold: self.permuted_ids[self.fold_size *
                                                             test_fold: self.fold_size * (1 + test_fold)] for test_fold in range(self.k)}
        self.partitions_train = {train_fold: np.array(
            [id_ for id_ in self.permuted_ids if id_ not in self.partitions_test[train_fold]]) for train_fold in range(self.k)}

    def get_tensor(self, df='visits_df', subset='train', fold=0):
        # e.g. dataset.visits_df_scaled_tensor_train[dataset.visits_df_proc[dataset.visits_df_proc.patient_id.isin(partition.partitions[0])].tensor_indices_train.values]
        tensor_name = df + '_scaled_tensor_train'
        tensor = getattr(self.dataset, tensor_name)
        processed_df = getattr(self.dataset, df + '_proc')
        if subset == 'test':
            tensor_slice = tensor[processed_df[processed_df.patient_id.isin(
                self.partitions_test[fold])].tensor_indices_train.values]
        else:
            tensor_slice = tensor[processed_df[processed_df.patient_id.isin(
                self.partitions_train[fold])].tensor_indices_train.values]
        return tensor_slice

    def set_current_fold(self, k):
        self.current_fold = k
