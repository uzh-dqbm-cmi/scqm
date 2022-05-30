import torch
import numpy as np
import matplotlib.pyplot as plt
import gc

from scqm.custom_library.trainers.trainer import Trainer
from scqm.custom_library.trainers.batch.batch import Batch
from scqm.custom_library.models.model import Model
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.partition.partition import DataPartition


class AdaptivenetTrainer(Trainer):
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
        super().__init__(model, dataset, n_epochs, batch_size, lr, use_early_stopping)
        self.balance_classes = balance_classes
        self.f1_per_epoch = torch.empty(size=(n_epochs, 1))
        self.f1_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        if model.task == "regression":
            self.criterion = torch.nn.MSELoss(reduction="sum")
        else:
            self.weights = self.get_weights(self.dataset)
            # self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.weights)
            self.criterion = torch.nn.CrossEntropyLoss(
                reduction="sum", weight=self.weights
            )

    def get_weights(self, dataset: Dataset) -> torch.tensor:
        """Compute weights to balance classes in classification task

        Args:
            dataset (Dataset): dataset

        Returns:
            torch.tensor: weights
        """
        if self.model.num_targets == 2:
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            class_0 = len(
                dataset.a_visit_df[patients_train][
                    dataset.a_visit_df[patients_train][dataset.target_category_name]
                    == 0
                ]
            )
            class_1 = len(
                dataset.a_visit_df[patients_train][
                    dataset.a_visit_df[patients_train][dataset.target_category_name]
                    == 1
                ]
            )
            # (the remaining are nan)
            weights = torch.tensor(class_1 / class_0) if self.balance_classes else None
        else:
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            weights = torch.empty(self.model.num_targets, device=self.model.device)
            for w in range(self.model.num_targets):
                weights[w] = len(
                    dataset.a_visit_df[patients_train][
                        dataset.a_visit_df[patients_train][dataset.target_category_name]
                        == w
                    ]
                )
            weights = weights / weights.sum()
            weights = 1 / weights if self.balance_classes else None
        return weights

    def train_model(self, model: Model, partition: DataPartition, debug_patient):
        print(f"device {model.device}")
        # dfs and tensors
        self.dataset.move_to_device(model.device)

        # validation
        # debug_patient_valid = np.random.choice(partition.partitions_test[partition.current_fold], size=1)[0]

        # print(
        #     f'Debug patient {debug_patient_valid} \nall targets \n{dataset.targets_df_proc[dataset.targets_df_proc.patient_id == debug_patient_valid]["das283bsr_score"]}')
        batch_valid = Batch(
            model.device,
            partition.partitions_test[partition.current_fold],
            partition.partitions_test[partition.current_fold],
            partition.partitions_test[partition.current_fold],
        )
        batch_valid.get_batch(self.dataset, debug_patient=None)
        batch_valid.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(
                partition.partitions_train[partition.current_fold], size=1
            )[0]
            print(f"complete timeline {self.dataset[debug_patient].timeline}")
            # debug_patient = '0034bdeb-8b71-15f5-2006-4a49be283b3f'
            print(
                f'Debug patient {debug_patient} \nall targets \n{self.dataset.targets_df_proc[self.dataset.targets_df_proc.patient_id == debug_patient][["das283bsr_score", self.dataset.target_category_name]]}'
            )
        # if model.task == 'classification':
        #     metrics = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0, 1, 2], device = model.device))
        # else :
        #     metrics = Metrics(device=model.device)

        batch = Batch(
            model.device,
            partition.partitions_train[partition.current_fold],
            partition.partitions_train[partition.current_fold],
        )
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            gc.collect()
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            batch.get_batch(self.dataset, self.batch_size, debug_patient)
            batch.get_masks(self.dataset, debug_patient)
            self.update_epoch_and_indices(batch)

            self.loss = model.apply_and_get_loss(self.dataset, self.criterion, batch)

            if self.loss:
                # take optimizer step once loss wrt all visits has been computed
                self.optimizer.zero_grad()

                self.loss.backward()
                self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.loss

                    model.eval()
                    # if model.task == 'classification':
                    #     metrics_val = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0,1,2], device = model.device))
                    # else:
                    #     metrics_val = Metrics(device=model.device)
                    self.loss_valid = model.apply_and_get_loss(
                        self.dataset, self.criterion, batch_valid
                    )
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid
                    # metrics.get_metrics()
                    # f1 = metrics.returned_metric
                    # self.f1_per_epoch[self.current_epoch - 1] = f1
                    # metrics_val.get_metrics()
                    # f1_val = metrics_val.returned_metric
                    # self.f1_per_epoch_valid[self.current_epoch - 1] = f1_val
                    # print(
                    #     f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid} f1 {f1} f1_valid {f1_val}')
                    # re-initialize metrics for new epoch
                    print(
                        f"epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}"
                    )
                    # if model.task == 'classification':
                    #     metrics = MulticlassMetrics(
                    #         device=model.device, possible_classes=torch.tensor([0, 1, 2], device=model.device))
                    # else:
                    #     metrics = Metrics(device=model.device)
                    if self.use_early_stopping:
                        self.check_early_stopping()
        return

    def plot_accuracy(self):
        plt.figure()
        plt.plot(
            range(0, len(self.accuracy_per_epoch[: self.current_epoch]), 1),
            self.accuracy_per_epoch[: self.current_epoch],
        )
        plt.plot(
            range(0, len(self.accuracy_per_epoch[: self.current_epoch]), 1),
            self.accuracy_per_epoch_valid[: self.current_epoch],
        )
        return
