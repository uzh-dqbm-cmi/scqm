from doctest import OutputChecker
from scqm.custom_library.trainers.trainer import Trainer
from scqm.custom_library.models.model import Model
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.partition.multitask_partition import MultitaskPartition
from scqm.custom_library.trainers.batch.batch import Batch
import timeit

import torch


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
            partition.partitions_train[partition.current_fold],
            partition.partitions_train[partition.current_fold],
            tensor_names=["joint_das28", "joint_targets_das28"],
            target_name="das283bsr_score",
        )
        train_tensor = self.dataset.joint_das28_df_scaled_tensor_train
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            # model.train()

            batch.get_batch(self.dataset, self.batch_size)
            self.update_epoch_and_indices(batch)

            output = model(train_tensor[batch.indices_joint_das28])
            # self.loss
            # if self.loss:
            #     # take optimizer step once loss wrt all visits has been computed
            #     self.optimizer.zero_grad()

            #     self.loss.backward()
            #     self.optimizer.step()

            # # store loss and evaluate on validation data
            # if len(batch.available_indices) == len(batch.all_indices):
            #     with torch.no_grad():
            #         self.loss_per_epoch[self.current_epoch - 1] = self.loss

            #         model.eval()
            #         # if model.task == 'classification':
            #         #     metrics_val = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0,1,2], device = model.device))
            #         # else:
            #         #     metrics_val = Metrics(device=model.device)
            #         self.loss_valid = model.apply_and_get_loss(
            #             self.dataset, self.criterion, batch_valid
            #         )
            #         self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid
            #         # metrics.get_metrics()
            #         # f1 = metrics.returned_metric
            #         # self.f1_per_epoch[self.current_epoch - 1] = f1
            #         # metrics_val.get_metrics()
            #         # f1_val = metrics_val.returned_metric
            #         # self.f1_per_epoch_valid[self.current_epoch - 1] = f1_val
            #         # print(
            #         #     f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid} f1 {f1} f1_valid {f1_val}')
            #         # re-initialize metrics for new epoch
            #         print(
            #             f"epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}"
            #         )
            #         # if model.task == 'classification':
            #         #     metrics = MulticlassMetrics(
            #         #         device=model.device, possible_classes=torch.tensor([0, 1, 2], device=model.device))
            #         # else:
            #         #     metrics = Metrics(device=model.device)
            #         if self.use_early_stopping:
            #             self.check_early_stopping()
        return
