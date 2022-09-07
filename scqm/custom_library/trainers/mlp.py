from doctest import OutputChecker
from scqm.custom_library.trainers.trainer import Trainer
from scqm.custom_library.models.model import Model
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.trainers.batch.batch import Batch
import timeit

import torch
import numpy as np


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

    def train_model(self, model: Model, partition: MultitaskPartition):
        print(f"model device {model.device}")
        # dfs and tensors
        self.dataset.move_to_device(model.device)
        batch_valid = Batch(
            model.device,
            partition.partitions_test_das28[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partition.partitions_test_das28[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partition.partitions_test_das28[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            tensor_names=["joint_das28", "joint_targets_das28"],
            target_name="das283bsr_score",
        )
        batch_valid.get_batch(self.dataset, debug_patient=None)
        batch = Batch(
            model.device,
            partition.partitions_train_das28[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            partition.partitions_train_das28[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            tensor_names=["joint_das28", "joint_targets_das28"],
            target_name="das283bsr_score",
        )
        train_indices = np.concatenate(
            [
                self.dataset.tensor_indices_mapping_train[patient]["joint_das28_df"]
                for patient in partition.partitions_train_das28[partition.current_fold]
                + partition.partitions_train_both[partition.current_fold]
            ]
        )
        train_tensor = self.dataset.joint_das28_df_scaled_tensor_train
        train_target = self.dataset.joint_targets_das28_df_scaled_tensor_train

        valid_tensor = self.dataset.joint_das28_df_scaled_tensor_train[
            batch_valid.indices_joint_das28
        ]
        valid_target = self.dataset.joint_targets_das28_df_scaled_tensor_train[
            batch_valid.indices_joint_targets_das28
        ]
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            model.train()

            batch.get_batch(self.dataset, self.batch_size)
            self.update_epoch_and_indices(batch)

            output = model(train_tensor[batch.indices_joint_das28])
            self.loss = self.criterion(
                train_target[batch.indices_joint_targets_das28], output
            )
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

                    print(
                        f"epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}"
                    )

        return
