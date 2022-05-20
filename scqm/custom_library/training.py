import torch
import numpy as np
from utils import MulticlassMetrics, Metrics
import matplotlib.pyplot as plt
import pandas as pd
import gc
import timeit

class Trainer:
    #TODO change dependencies, pass e.g. n_epochs as param to train
    def __init__(self, model, dataset, n_epochs, batch_size, lr, use_early_stopping):
        self.dataset = dataset
        self.n_epochs = n_epochs
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        # use early stopping or not
        self.use_early_stopping = use_early_stopping
        # flag used during training to indicate whether to stop or not
        self.early_stopping = False
        #self.optimizer = torch.optim.AdamW(model.parameters, lr=self.lr, weight_decay=0.07)
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
        # TODO check that and implement something like https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
        if self.current_epoch > 35 and (self.loss_per_epoch_valid[self.current_epoch - 1 - 20:self.current_epoch - 1] / self.loss_per_epoch_valid[self.current_epoch - 1 - 30:self.current_epoch - 1 - 10]).mean() > 1:
            self.early_stopping = True
            self.loss_per_epoch = self.loss_per_epoch[:self.current_epoch - 1]
            self.loss_per_epoch_valid = self.loss_per_epoch_valid[:self.current_epoch - 1]
            print('early stopping because validation loss is increasing')

    def update_epoch_and_indices(self, batch):
        # remove batch indices from available indices (since one epoch is one pass through whole data set)
        batch.available_indices = np.array([x for x in batch.available_indices if x not in batch.current_indices])
        # a whole pass through the data has been completed
        if len(batch.available_indices) == 0:
            self.stop = timeit.default_timer()
            print(f'Time: for epoch {(self.stop - self.start)/60} total time {(self.stop - self.total_time)/60}')
            
            self.start = timeit.default_timer()
            batch.available_indices = batch.all_indices
            self.current_epoch += 1
        return

    def train_model(self):
        raise NotImplementedError

    def plot_losses(self):
        plt.figure()
        plt.plot(range(0, len(self.loss_per_epoch[:self.current_epoch]), 1), self.loss_per_epoch[:self.current_epoch])
        plt.plot(range(0, len(self.loss_per_epoch[:self.current_epoch]), 1),
                 self.loss_per_epoch_valid[:self.current_epoch])
        return


class AdaptivenetTrainer(Trainer):
    def __init__(self, model, dataset, n_epochs, batch_size, lr, balance_classes, use_early_stopping):
        super().__init__(model, dataset, n_epochs, batch_size, lr, use_early_stopping)
        self.balance_classes = balance_classes
        self.f1_per_epoch = torch.empty(size=(n_epochs, 1))
        self.f1_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        if model.task == 'regression':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        else:
            self.weights = self.get_weights(self.dataset)
            #self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.weights)
            self.criterion = torch.nn.CrossEntropyLoss(reduction='sum', weight=self.weights)

    def get_weights(self, dataset):
        if self.model.num_targets == 2:
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            class_0 = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                            [dataset.target_category_name] == 0])
            class_1 = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                            [dataset.target_category_name] == 1])
            # (the remaining are nan)
            weights = torch.tensor(class_1 / class_0) if self.balance_classes else None
        else :
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            weights = torch.empty(self.model.num_targets, device = self.model.device)
            for w in range(self.model.num_targets):
                weights[w] = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                                    [dataset.target_category_name] == w])
            weights = weights / weights.sum()
            weights = 1 / weights if self.balance_classes else None
        return weights
    

    def train_model(self, model, partition, debug_patient):
        print(f'device {model.device}')
        #dfs and tensors
        self.dataset.move_to_device(model.device)

        # validation
        # debug_patient_valid = np.random.choice(partition.partitions_test[partition.current_fold], size=1)[0]

        # print(
        #     f'Debug patient {debug_patient_valid} \nall targets \n{dataset.targets_df_proc[dataset.targets_df_proc.patient_id == debug_patient_valid]["das283bsr_score"]}')
        batch_valid = Batch(model.device, partition.partitions_test[partition.current_fold],
                            partition.partitions_test[partition.current_fold], partition.partitions_test[partition.current_fold])
        batch_valid.get_batch(self.dataset, debug_patient=None)
        batch_valid.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(partition.partitions_train[partition.current_fold], size=1)[0]
            print(f'complete timeline {self.dataset[debug_patient].timeline}')
            #debug_patient = '0034bdeb-8b71-15f5-2006-4a49be283b3f'
            print(
                f'Debug patient {debug_patient} \nall targets \n{self.dataset.targets_df_proc[self.dataset.targets_df_proc.patient_id == debug_patient][["das283bsr_score", self.dataset.target_category_name]]}')
        # if model.task == 'classification':
        #     metrics = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0, 1, 2], device = model.device))
        # else :
        #     metrics = Metrics(device=model.device)

        batch = Batch(model.device, partition.partitions_train[partition.current_fold],
                      partition.partitions_train[partition.current_fold])
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            gc.collect()
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            batch.get_batch(self.dataset, self.batch_size, debug_patient)
            batch.get_masks(
                self.dataset, debug_patient)
            self.update_epoch_and_indices(batch)
      
            self.loss = model.apply_and_get_loss(self.dataset, self.criterion, batch)


     
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
                    self.loss_valid = model.apply_and_get_loss(self.dataset, self.criterion,
                                                                            batch_valid)
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
                    print(f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}')
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
        plt.plot(range(0, len(self.accuracy_per_epoch[:self.current_epoch]),
                 1), self.accuracy_per_epoch[:self.current_epoch])
        plt.plot(range(0, len(self.accuracy_per_epoch[:self.current_epoch]), 1),
                 self.accuracy_per_epoch_valid[:self.current_epoch])
        return


class AutoEncoderAdaptivenetTrainer(Trainer):
    def __init__(self, model, dataset, n_epochs, batch_size, lr, balance_classes, use_early_stopping):
        super().__init__(model, dataset, n_epochs, batch_size, lr, use_early_stopping)
        self.balance_classes = balance_classes
        self.f1_per_epoch = torch.empty(size=(n_epochs, 1))
        self.f1_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        self.dec_criterion = torch.nn.MSELoss()
        self.loss_per_epoch_decod = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_decod_valid = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_pred = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_pred_valid = torch.empty(size=(n_epochs, 1))
        if model.task == 'regression':
            self.pred_criterion = torch.nn.MSELoss(reduction='sum')
        else:
            self.weights = self.get_weights(self.dataset)
            #self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=self.weights)
            self.pred_criterion = torch.nn.CrossEntropyLoss(reduction='sum', weight=self.weights)

    def get_weights(self, dataset):
        if self.model.num_targets == 2:
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            class_0 = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                             [dataset.target_category_name] == 0])
            class_1 = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                             [dataset.target_category_name] == 1])
            # (the remaining are nan)
            weights = torch.tensor(class_1 / class_0) if self.balance_classes else None
        else:
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            weights = torch.empty(self.model.num_targets, device=self.model.device)
            for w in range(self.model.num_targets):
                weights[w] = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                                    [dataset.target_category_name] == w])
            weights = weights / weights.sum()
            weights = 1 / weights if self.balance_classes else None
        return weights

    def train_model(self, model, partition, debug_patient):
        print(f'device {model.device}')
        #dfs and tensors
        self.dataset.move_to_device(model.device)

        # validation
        # debug_patient_valid = np.random.choice(partition.partitions_test[partition.current_fold], size=1)[0]

        # print(
        #     f'Debug patient {debug_patient_valid} \nall targets \n{dataset.targets_df_proc[dataset.targets_df_proc.patient_id == debug_patient_valid]["das283bsr_score"]}')
        batch_valid = Batch(model.device, partition.partitions_test[partition.current_fold],
                            partition.partitions_test[partition.current_fold], partition.partitions_test[partition.current_fold])
        batch_valid.get_batch(self.dataset, debug_patient=None)
        batch_valid.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(partition.partitions_train[partition.current_fold], size=1)[0]
            #debug_patient = '0034bdeb-8b71-15f5-2006-4a49be283b3f'
            print(
                f'Debug patient {debug_patient} \nall targets \n{self.dataset.targets_df_proc[self.dataset.targets_df_proc.patient_id == debug_patient][["das283bsr_score", self.dataset.target_category_name]]}')
        # if model.task == 'classification':
        #     metrics = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0, 1, 2], device = model.device))
        # else :
        #     metrics = Metrics(device=model.device)

        batch = Batch(model.device, partition.partitions_train[partition.current_fold],
                      partition.partitions_train[partition.current_fold])
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            gc.collect()
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            batch.get_batch(self.dataset, self.batch_size, debug_patient)
            batch.get_masks(
                self.dataset, debug_patient)
            self.update_epoch_and_indices(batch)
            self.decod_loss, self.pred_loss = model.apply_and_get_loss(self.dataset, self.dec_criterion, self.pred_criterion, batch)

            # take optimizer step once loss wrt all visits has been computed
            self.optimizer.zero_grad()
            (self.decod_loss + self.pred_loss).backward()
            self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.decod_loss + self.pred_loss
                    self.loss_per_epoch_decod[self.current_epoch - 1] = self.decod_loss
                    self.loss_per_epoch_pred[self.current_epoch - 1] = self.pred_loss
                    model.eval()
                    # if model.task == 'classification':
                    #     metrics_val = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0,1,2], device = model.device))
                    # else:
                    #     metrics_val = Metrics(device=model.device)
                    self.decod_loss_valid, self.pred_loss_valid = model.apply_and_get_loss(self.dataset, self.dec_criterion, self.pred_criterion,
                                                               batch_valid)
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.decod_loss_valid + self.pred_loss_valid
                    self.loss_per_epoch_decod_valid[self.current_epoch - 1] = self.decod_loss_valid 
                    self.loss_per_epoch_pred_valid[self.current_epoch - 1] =  self.pred_loss_valid
                    # metrics.get_metrics()
                    # f1 = metrics.returned_metric
                    # self.f1_per_epoch[self.current_epoch - 1] = f1
                    # metrics_val.get_metrics()
                    # f1_val = metrics_val.returned_metric
                    # self.f1_per_epoch_valid[self.current_epoch - 1] = f1_val
                    # print(
                    #     f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid} f1 {f1} f1_valid {f1_val}')
                    # re-initialize metrics for new epoch
                    print(f'epoch : {self.current_epoch} pred loss {self.pred_loss} loss_valid {self.pred_loss_valid}')
                    print(f'epoch : {self.current_epoch} decod loss {self.decod_loss} loss_valid {self.decod_loss_valid}')
                    # if model.task == 'classification':
                    #     metrics = MulticlassMetrics(
                    #         device=model.device, possible_classes=torch.tensor([0, 1, 2], device=model.device))
                    # else:
                    #     metrics = Metrics(device=model.device)
                    if self.use_early_stopping:
                        self.check_early_stopping()
        return

    def plot_other_losses(self):
        plt.figure()
        plt.plot(range(0, len(self.loss_per_epoch_decod[:self.current_epoch]),
                 1), self.loss_per_epoch_decod[:self.current_epoch])
        plt.plot(range(0, len(self.loss_per_epoch_decod[:self.current_epoch]), 1),
                 self.loss_per_epoch_decod_valid[:self.current_epoch])
        plt.figure()
        plt.plot(range(0, len(self.loss_per_epoch_pred[:self.current_epoch]),
                 1), self.loss_per_epoch_pred[:self.current_epoch])
        plt.plot(range(0, len(self.loss_per_epoch_pred[:self.current_epoch]), 1),
                 self.loss_per_epoch_pred_valid[:self.current_epoch])
        return 

class AutoEncoderTrainer(Trainer):
    def __init__(self, model, dataset, n_epochs, batch_size, lr, use_early_stopping):
        super().__init__(model, dataset, n_epochs, batch_size, lr, use_early_stopping)
        self.f1_per_epoch = torch.empty(size=(n_epochs, 1))
        self.f1_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        self.criterion = torch.nn.MSELoss()


    def train_model(self, model, partition, debug_patient):
        print(f'device {model.device}')
        #dfs and tensors
        self.dataset.move_to_device(model.device)

        # validation
        # debug_patient_valid = np.random.choice(partition.partitions_test[partition.current_fold], size=1)[0]

        # print(
        #     f'Debug patient {debug_patient_valid} \nall targets \n{dataset.targets_df_proc[dataset.targets_df_proc.patient_id == debug_patient_valid]["das283bsr_score"]}')
        batch_valid = Batch(model.device, partition.partitions_test[partition.current_fold],
                            partition.partitions_test[partition.current_fold], partition.partitions_test[partition.current_fold])
        batch_valid.get_batch(self.dataset, debug_patient=None)
        #batch_valid.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(partition.partitions_train[partition.current_fold], size=1)[0]
            #debug_patient = '0034bdeb-8b71-15f5-2006-4a49be283b3f'
            print(
                f'Debug patient {debug_patient} \nall targets \n{self.dataset.targets_df_proc[self.dataset.targets_df_proc.patient_id == debug_patient][["das283bsr_score", self.dataset.target_category_name]]}')
        # if model.task == 'classification':
        #     metrics = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0, 1, 2], device = model.device))
        # else :
        #     metrics = Metrics(device=model.device)

        batch = Batch(model.device, partition.partitions_train[partition.current_fold],
                      partition.partitions_train[partition.current_fold])
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            gc.collect()
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            batch.get_batch(self.dataset, self.batch_size, debug_patient)
            # batch.get_masks(
            #     self.dataset, debug_patient)
            self.update_epoch_and_indices(batch)
            self.loss, self.all_losses = model.apply_and_get_loss(self.dataset, self.criterion, batch)

            # take optimizer step once loss wrt all visits has been computed
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.loss

                    model.eval()
                    self.loss_valid, self.all_losses_valid = model.apply_and_get_loss(self.dataset, self.criterion,
                                                               batch_valid)
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid

                    # re-initialize metrics for new epoch

                    print(f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid}')

                    if self.use_early_stopping:
                        self.check_early_stopping()
        return


class Batch:
    def __init__(self, device, all_indices, available_indices, current_indices=None):
        self.device = device
        self.all_indices = all_indices
        self.available_indices = available_indices
        self.current_indices = current_indices

    def get_batch(self, dataset, batch_size=None, debug_patient=None):
        """ First, selects a batch of patients from the available indices for this epoch and the corresponding tensor (visits/medications/
        patient) slices. Then for each visit v, for each patient of the batch, create a mask to combine the visits/medication events coming before v in the right order. 

        """
        # during training, only select subset of available indices
        if batch_size:
            # batch size
            size = min(len(self.available_indices), batch_size)
            # batch and corresponding tensor indices
            self.current_indices = np.random.choice(self.available_indices, size=size, replace=False)
            #print(f'len available indices {len(self.available_indices)}')

        for name in dataset.event_names + ['patients', 'targets']:
            indices= [dataset.tensor_indices_mapping_train[patient][name + '_df'] for patient in self.current_indices]
            if len(indices) > 0:
                indices = np.concatenate(indices)
            setattr(self, 'indices_' + name,
                    indices)
        if debug_patient and debug_patient in self.current_indices:
            self.debug_index = list(self.current_indices).index(debug_patient)
            print(f'index in batch {self.debug_index}')
        else:
            self.debug_index = None
        if (self.indices_a_visit != self.indices_targets).any():
            raise ValueError('index mismatch between visits and targets')

        return

    def get_masks(self, dataset, debug_patient):
        """_summary_

        Args:
            dataset (_type_): _description_
            subset (_type_): _description_
            min_num_visits (_type_): min number of initial visits to retrieve the information from
        e.g. if min_num_visits = 2, for each patient we start retrieving all information
        up to the 2nd visit, i.e. medications before 2nd visit and info about 1st visit
        (in other words, min_num_visits is the first target visit). For each visit v >= min_num_visits, we store for each patient the number of visits and medication events
        up to v

        Returns:
            _type_: _description_
        """
        
        indices_mapping = [dataset.mapping_for_masks[id_] for id_ in self.current_indices]
        self.seq_lengths = dataset.masks.seq_lengths[:,indices_mapping, :]
        for event in dataset.event_names:
            name = event + '_masks'
            setattr(self, name, list(getattr(dataset.masks, name)[i] for i in indices_mapping))
        self.available_visit_mask = dataset.masks.available_visit_mask[indices_mapping]
        self.total_num = dataset.masks.total_num[indices_mapping]
        self.max_num_visits = max(list(dataset.masks.num_visits[i] for i in indices_mapping))

        if debug_patient and debug_patient in self.current_indices:
            index = dataset.mapping_for_masks[debug_patient]
            for visit in range(dataset.masks.num_visits[index] - dataset.min_num_visits + 1):
                _, _, _, visual = dataset.patients[debug_patient].get_cropped_timeline(
                    visit + dataset.min_num_visits)
                print(
                    f'visit {visit} cropped timeline mask {visual}')
    
        return




