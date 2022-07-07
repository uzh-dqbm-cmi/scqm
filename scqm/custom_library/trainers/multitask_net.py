from scqm.custom_library.trainers.adaptive_net import AdaptivenetTrainer
import torch
import numpy as np
from scqm.custom_library.trainers.trainer import Trainer
from scqm.custom_library.trainers.batch.batch import Batch
from scqm.custom_library.models.model import Model
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
import matplotlib.pyplot as plt
import timeit


class MultitaskTrainer(AdaptivenetTrainer):
    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        n_epochs: int,
        batch_size: int,
        lr: float,
        balance_classes: bool,
        use_early_stopping: bool,
    ):
        """Instanciate trainer

        Args:
            model (Model): model to train
            dataset (Dataset): dataset
            n_epochs (int): number of epochs
            batch_size (int): batch size
            lr (float): _description_
            balance_classes (bool): for classification, whether to use weight to balance target classes
            use_early_stopping (bool): stop when validation loss is increasing
        """
        super().__init__(
            model,
            dataset,
            n_epochs,
            batch_size,
            lr,
            balance_classes,
            use_early_stopping,
        )
        self.loss_per_epoch_das28 = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_valid_das28 = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_basdai = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_valid_basdai = torch.empty(size=(n_epochs, 1))
        self.batch_size_das28 = batch_size["das28"]
        self.batch_size_basdai = batch_size["basdai"]

    def update_epoch_and_indices(self, batches: list):
        """Update available indices and epoch if a pass has been made through all the data

        Args:
            batch (Batch): current batch
        """
        # remove batch indices from available indices (since one epoch is one pass through whole data set)
        for batch in batches:
            batch.available_indices = np.array(
                [x for x in batch.available_indices if x not in batch.current_indices]
            )
        # a whole pass through the data has been completed
        if sum([len(batch.available_indices) for batch in batches]) == 0:
            self.stop = timeit.default_timer()
            print(
                f"Time: for epoch {(self.stop - self.start)/60} total time {(self.stop - self.total_time)/60}"
            )

            self.start = timeit.default_timer()
            for batch in batches:
                batch.available_indices = batch.all_indices
            self.current_epoch += 1
        return

    def train_model(self, model: Model, partition: MultitaskPartition, debug_patient):
        print(f"device {model.device}")
        # dfs and tensors
        self.dataset.move_to_device(model.device)
        # create separate batches for both types of targets
        batch_valid_das28 = Batch(
            model.device,
            partition.partitions_test_das28[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partition.partitions_test_das28[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partition.partitions_test_das28[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            tensor_names=self.dataset.event_names
            + ["patients", "targets_das28", "targets_basdai"],
            target_name="das283bsr_score",
        )
        batch_valid_das28.get_batch(self.dataset, debug_patient=None)
        batch_valid_das28.get_masks(self.dataset, debug_patient=None)

        batch_valid_basdai = Batch(
            model.device,
            partition.partitions_test_basdai[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partition.partitions_test_basdai[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            partition.partitions_test_basdai[partition.current_fold]
            + partition.partitions_test_both[partition.current_fold],
            tensor_names=self.dataset.event_names
            + ["patients", "targets_das28", "targets_basdai"],
            target_name="basdai_score",
        )
        batch_valid_basdai.get_batch(self.dataset, debug_patient=None)
        batch_valid_basdai.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(
                partition.partitions_train[partition.current_fold], size=1
            )[0]
            print(f"complete timeline {self.dataset[debug_patient].timeline}")
            print(
                f'Debug patient {debug_patient} \nall targets \n{self.dataset.targets_df_proc[self.dataset.targets_df_proc.patient_id == debug_patient][["das283bsr_score", self.dataset.target_category_name]]}'
            )

        batch_das28 = Batch(
            model.device,
            partition.partitions_train_das28[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            partition.partitions_train_das28[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            tensor_names=self.dataset.event_names
            + ["patients", "targets_das28", "targets_basdai"],
            target_name="das283bsr_score",
            special_indices=partition.partitions_train_both[partition.current_fold],
        )

        batch_basdai = Batch(
            model.device,
            partition.partitions_train_basdai[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            partition.partitions_train_basdai[partition.current_fold]
            + partition.partitions_train_both[partition.current_fold],
            tensor_names=self.dataset.event_names
            + ["patients", "targets_das28", "targets_basdai"],
            target_name="basdai_score",
            special_indices=partition.partitions_train_both[partition.current_fold],
        )
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            # gc.collect()
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            indices_to_include = batch_das28.get_batch(
                self.dataset, self.batch_size_das28, debug_patient
            )
            batch_das28.get_masks(self.dataset, debug_patient)
            if indices_to_include is None:
                indices_to_exclude = batch_das28.special_indices
            else:
                indices_to_exclude = [
                    index
                    for index in batch_das28.special_indices
                    if index not in indices_to_include
                ]
            batch_basdai.get_batch(
                self.dataset,
                self.batch_size_basdai,
                debug_patient,
                indices_to_include=indices_to_include,
                indices_to_exclude=indices_to_exclude,
            )
            batch_basdai.get_masks(self.dataset, debug_patient)
            self.update_epoch_and_indices([batch_das28, batch_basdai])

            self.loss = 0
            if len(batch_das28.current_indices) > 0:
                self.loss_das28 = model.apply_and_get_loss(
                    self.dataset, self.criterion, batch_das28, "das283bsr_score"
                )
                self.loss += self.loss_das28
            if len(batch_basdai.current_indices) > 0:
                self.loss_basdai = model.apply_and_get_loss(
                    self.dataset, self.criterion, batch_basdai, "basdai_score"
                )
                self.loss += self.loss_basdai
            if self.loss:
                # take optimizer step once loss wrt all visits has been computed
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch_basdai.available_indices) == len(batch_basdai.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = (
                        self.loss_das28 + self.loss_basdai
                    )
                    self.loss_per_epoch_das28[self.current_epoch - 1] = self.loss_das28
                    self.loss_per_epoch_basdai[
                        self.current_epoch - 1
                    ] = self.loss_basdai
                    model.eval()
                    # if model.task == 'classification':
                    #     metrics_val = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0,1,2], device = model.device))
                    # else:
                    #     metrics_val = Metrics(device=model.device)
                    self.loss_valid_das28 = model.apply_and_get_loss(
                        self.dataset,
                        self.criterion,
                        batch_valid_das28,
                        "das283bsr_score",
                    )
                    self.loss_valid_basdai = model.apply_and_get_loss(
                        self.dataset, self.criterion, batch_valid_basdai, "basdai_score"
                    )
                    self.loss_valid = self.loss_valid_das28 + self.loss_valid_basdai
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid
                    self.loss_per_epoch_valid_das28[
                        self.current_epoch - 1
                    ] = self.loss_valid_das28
                    self.loss_per_epoch_valid_basdai[
                        self.current_epoch - 1
                    ] = self.loss_valid_basdai
                    print(
                        f"epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}"
                        f"loss das28 {self.loss_das28} loss_valid {self.loss_valid_das28}"
                        f"loss basdai {self.loss_basdai} loss_valid {self.loss_valid_basdai}"
                    )

                    if self.use_early_stopping:
                        self.check_early_stopping()
        return

    def plot_losses(self):
        plt.figure()
        plt.plot(
            range(0, len(self.loss_per_epoch[: self.current_epoch]), 1),
            self.loss_per_epoch[: self.current_epoch],
        )
        plt.plot(
            range(0, len(self.loss_per_epoch[: self.current_epoch]), 1),
            self.loss_per_epoch_valid[: self.current_epoch],
        )
        plt.figure()
        plt.plot(
            range(0, len(self.loss_per_epoch_das28[: self.current_epoch]), 1),
            self.loss_per_epoch_das28[: self.current_epoch],
        )
        plt.plot(
            range(0, len(self.loss_per_epoch[: self.current_epoch]), 1),
            self.loss_per_epoch_valid_das28[: self.current_epoch],
        )
        plt.figure()
        plt.plot(
            range(0, len(self.loss_per_epoch[: self.current_epoch]), 1),
            self.loss_per_epoch_basdai[: self.current_epoch],
        )
        plt.plot(
            range(0, len(self.loss_per_epoch[: self.current_epoch]), 1),
            self.loss_per_epoch_valid_basdai[: self.current_epoch],
        )
