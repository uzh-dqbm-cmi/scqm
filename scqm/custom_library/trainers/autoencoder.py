from scqm.custom_library.trainers.trainer import Trainer
import torch
import numpy as np
import gc
from scqm.custom_library.trainers.trainer import Trainer
from scqm.custom_library.trainers.batch.batch import Batch

class AutoEncoderTrainer(Trainer):
    def __init__(self, model, dataset, n_epochs, batch_size, lr, use_early_stopping):
        super().__init__(model, dataset, n_epochs, batch_size, lr, use_early_stopping)
        self.f1_per_epoch = torch.empty(size=(n_epochs, 1))
        self.f1_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        self.criterion = torch.nn.MSELoss()

    def train_model(self, model, partition, debug_patient):
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
        # batch_valid.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(
                partition.partitions_train[partition.current_fold], size=1
            )[0]
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
            # batch.get_masks(
            #     self.dataset, debug_patient)
            self.update_epoch_and_indices(batch)
            self.loss, self.all_losses = model.apply_and_get_loss(
                self.dataset, self.criterion, batch
            )

            # take optimizer step once loss wrt all visits has been computed
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.loss

                    model.eval()
                    self.loss_valid, self.all_losses_valid = model.apply_and_get_loss(
                        self.dataset, self.criterion, batch_valid
                    )
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid

                    # re-initialize metrics for new epoch

                    print(
                        f"epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}"
                    )

                    if self.use_early_stopping:
                        self.check_early_stopping()
        return
