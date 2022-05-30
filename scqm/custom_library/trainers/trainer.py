import torch
import timeit
import matplotlib.pyplot as plt
import numpy as np

from scqm.custom_library.models.model import Model
from scqm.custom_library.data_objects.dataset import Dataset
from scqm.custom_library.trainers.batch.batch import Batch


class Trainer:
    """Base trainer object class"""

    # TODO change dependencies, pass e.g. n_epochs as param to train
    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        n_epochs: int,
        batch_size: int,
        lr: float,
        use_early_stopping: bool,
    ):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        # use early stopping or not
        self.use_early_stopping = use_early_stopping
        # flag used during training to indicate whether to stop or not
        self.early_stopping = False
        # self.optimizer = torch.optim.AdamW(model.parameters, lr=self.lr, weight_decay=0.07)
        self.optimizer = torch.optim.AdamW(model.parameters, lr=self.lr)
        self.current_epoch = 0
        self.loss_per_epoch = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        self.start = timeit.default_timer()
        self.total_time = timeit.default_timer()
        # if model.task == 'regression':
        #     self.criterion = torch.nn.MSELoss(reduction='sum')
        # else:
        #     self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def check_early_stopping(self):
        """check if validation loss is increasing"""
        # TODO check that and implement something like https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        if (
            self.current_epoch > 35
            and (
                self.loss_per_epoch_valid[
                    self.current_epoch - 1 - 20 : self.current_epoch - 1
                ]
                / self.loss_per_epoch_valid[
                    self.current_epoch - 1 - 30 : self.current_epoch - 1 - 10
                ]
            ).mean()
            > 1
        ):
            self.early_stopping = True
            self.loss_per_epoch = self.loss_per_epoch[: self.current_epoch - 1]
            self.loss_per_epoch_valid = self.loss_per_epoch_valid[
                : self.current_epoch - 1
            ]
            print("early stopping because validation loss is increasing")

    def update_epoch_and_indices(self, batch: Batch):
        """Update available indices and epoch if a pass has been made through all the data

        Args:
            batch (Batch): current batch
        """
        # remove batch indices from available indices (since one epoch is one pass through whole data set)
        batch.available_indices = np.array(
            [x for x in batch.available_indices if x not in batch.current_indices]
        )
        # a whole pass through the data has been completed
        if len(batch.available_indices) == 0:
            self.stop = timeit.default_timer()
            print(
                f"Time: for epoch {(self.stop - self.start)/60} total time {(self.stop - self.total_time)/60}"
            )

            self.start = timeit.default_timer()
            batch.available_indices = batch.all_indices
            self.current_epoch += 1
        return

    def train_model(self):
        raise NotImplementedError

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
        return
