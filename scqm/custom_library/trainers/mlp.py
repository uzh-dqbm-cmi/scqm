from doctest import OutputChecker
from scqm.custom_library.trainers.trainer import Trainer
from scqm.custom_library.models.model import Model
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.trainers.batch.batch import Batch
import timeit

import torch
import numpy as np
import copy


class MLPTrainer(Trainer):
    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        n_epochs: int,
        batch_size: int,
        lr: float,
        balance_classes: bool,
        use_early_stopping: bool,
        target_name: str,
    ):
        """Instantiate trainer

        Args:
            model (Model): model to train
            dataset (Dataset): dataset
            n_epochs (int): number of epochs
            batch_size (int): batch size
            lr (float): _description_
            balance_classes (bool): for classification, whether to use weight to balance target classes
            use_early_stopping (bool): stop when validation loss is increasing
        """
        self.model = model
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.current_epoch = 0
        self.early_stopping = use_early_stopping
        self.target_name = target_name
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        self.loss_per_epoch = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        self.start = timeit.default_timer()
        self.total_time = timeit.default_timer()
        self.criterion = torch.nn.MSELoss()

    def train_model(self, model: Model, partition: MultitaskPartition, verbose=True):
        print(f"model device {model.device}")
        # dfs and tensors
        self.dataset.move_to_device(model.device)
        if self.target_name == "das283bsr_score":
            partitions_test = partition.partitions_test_das28
            partitions_train = partition.partitions_train_das28
            tensor_names = ["joint_das28", "joint_targets_das28"]
            train_tensor = self.dataset.joint_das28_df_scaled_tensor_train
            train_target = self.dataset.joint_targets_das28_df_scaled_tensor_train
        elif self.target_name == "asdas_score":
            partitions_test = partition.partitions_test_asdas
            partitions_train = partition.partitions_train_asdas
            tensor_names = ["joint_asdas", "joint_targets_asdas"]
            train_tensor = self.dataset.joint_asdas_df_scaled_tensor_train
            train_target = self.dataset.joint_targets_asdas_df_scaled_tensor_train
        batch_valid = Batch(
            model.device,
            partitions_test[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partitions_test[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partitions_test[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            tensor_names=tensor_names,
            target_name=self.target_name,
        )
        batch_valid.get_batch(self.dataset, debug_patient=None)
        batch = Batch(
            model.device,
            partitions_train[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            partitions_train[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            tensor_names=tensor_names,
            target_name=self.target_name,
        )
        # train_indices = np.concatenate(
        #     [
        #         self.dataset.tensor_indices_mapping_train[patient]["joint_das28_df"]
        #         for patient in partition.partitions_train_das28[partition.current_fold]
        #         + partition.partitions_train_both[partition.current_fold]
        #     ]
        # )
        if self.target_name == "das283bsr_score":
            valid_tensor = self.dataset.joint_das28_df_scaled_tensor_train[
                batch_valid.indices_joint_das28
            ]
            valid_target = self.dataset.joint_targets_das28_df_scaled_tensor_train[
                batch_valid.indices_joint_targets_das28
            ]
        elif self.target_name == "asdas_score":
            valid_tensor = self.dataset.joint_asdas_df_scaled_tensor_train[
                batch_valid.indices_joint_asdas
            ]
            valid_target = self.dataset.joint_targets_asdas_df_scaled_tensor_train[
                batch_valid.indices_joint_targets_asdas
            ]
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            model.train()

            batch.get_batch(self.dataset, self.batch_size)
            self.update_epoch_and_indices(batch)
            if self.target_name == "das283bsr_score":
                batch_indices = batch.indices_joint_das28
                batch_indices_targets = batch.indices_joint_targets_das28
            elif self.target_name == "asdas_score":
                batch_indices = batch.indices_joint_asdas
                batch_indices_targets = batch.indices_joint_targets_asdas
            output = model(train_tensor[batch_indices])
            self.loss = self.criterion(train_target[batch_indices_targets], output)
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()
            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.loss

                    model.eval()
                    out_valid = model(valid_tensor)
                    self.loss_valid = self.criterion(out_valid, valid_target)
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid

                    if verbose:
                        print(
                            f"epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}"
                        )

                    if self.current_epoch == 1:
                        self.best_model = copy.deepcopy(self.model)
                        self.best_loss_valid = self.loss_valid
                        self.optimal_epoch = self.current_epoch
                    elif self.loss_valid < self.best_loss_valid:
                        self.best_model = copy.deepcopy(self.model)
                        self.best_loss_valid = self.loss_valid
                        self.optimal_epoch = self.current_epoch

        return self.best_loss_valid.item()
