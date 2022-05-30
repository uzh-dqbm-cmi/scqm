import math
import numpy as np
from scqm.custom_library.data_objects.dataset import Dataset


class DataPartition:
    def __init__(self, dataset: Dataset, k: int = 5):
        """Instantiate data partition object and create folds

        Args:
            dataset (Dataset): dataset
            k (int, optional): Number of folds. Defaults to 5.
        """
        self.dataset = dataset
        self.k = k
        self.split()

    def split(self):
        """create folds"""
        # TODO implement stratifier (on the targets)
        # split data into train and test (no valid)
        self.train_ids, self.valid_ids, self.test_ids = self.dataset.split_data(
            prop_valid=0.0, prop_test=0.2
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

    def set_current_fold(self, k: int):
        """Keep track of current fold

        Args:
            k (int): current fold
        """
        self.current_fold = k
