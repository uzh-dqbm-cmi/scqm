from scqm.custom_library.partition.partition import DataPartition
import math
import numpy as np
from scqm.custom_library.data_objects.dataset import Dataset


class MultitaskPartition(DataPartition):
    def __init__(self, dataset: Dataset, k: int = 5, split_indices: dict = {}):
        self.dataset = dataset
        self.k = k
        self.split(ids=split_indices)

    def split(self, ids={}):
        """create folds"""
        # TODO implement stratifier (on the targets)
        # split data into train and test (no valid)
        if len(ids) == 0:
            self.train_ids, self.valid_ids, self.test_ids = self.dataset.split_data(
                prop_valid=0.0, prop_test=0.2
            )
        else:
            self.dataset.train_ids, self.dataset.valid_ids, self.dataset.test_ids = (
                ids["train_ids"],
                ids["valid_ids"],
                ids["test_ids"],
            )
            self.train_ids, self.valid_ids, self.test_ids = (
                ids["train_ids"],
                ids["valid_ids"],
                ids["test_ids"],
            )
        self.dataset.scale_and_tensor()
        # get partition of size k of train set
        self.fold_size = math.ceil(len(self.dataset.train_ids) / self.k)
        self.permuted_ids = np.random.permutation(self.dataset.train_ids)
        self.partitions_test = {
            test_fold: self.permuted_ids[
                self.fold_size * test_fold : self.fold_size * (1 + test_fold)
            ]
            for test_fold in range(self.k)
        }
        self.partitions_train = {
            train_fold: np.array(
                [
                    id_
                    for id_ in self.permuted_ids
                    if id_ not in self.partitions_test[train_fold]
                ]
            )
            for train_fold in range(self.k)
        }
        self.partitions_test_das28 = {
            test_fold: [
                patient
                for patient in self.partitions_test[test_fold]
                if self.dataset[patient].target_name == "das283bsr_score"
            ]
            for test_fold in range(self.k)
        }
        self.partitions_test_asdas = {
            test_fold: [
                patient
                for patient in self.partitions_test[test_fold]
                if self.dataset[patient].target_name == "asdas_score"
            ]
            for test_fold in range(self.k)
        }
        self.partitions_test_both = {
            test_fold: [
                patient
                for patient in self.partitions_test[test_fold]
                if self.dataset[patient].target_name == "both"
            ]
            for test_fold in range(self.k)
        }
        self.partitions_train_das28 = {
            train_fold: [
                patient
                for patient in self.partitions_train[train_fold]
                if self.dataset[patient].target_name == "das283bsr_score"
            ]
            for train_fold in range(self.k)
        }
        self.partitions_train_asdas = {
            train_fold: [
                patient
                for patient in self.partitions_train[train_fold]
                if self.dataset[patient].target_name == "asdas_score"
            ]
            for train_fold in range(self.k)
        }
        self.partitions_train_both = {
            train_fold: [
                patient
                for patient in self.partitions_train[train_fold]
                if self.dataset[patient].target_name == "both"
            ]
            for train_fold in range(self.k)
        }
