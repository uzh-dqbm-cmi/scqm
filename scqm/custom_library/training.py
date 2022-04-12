import torch
import numpy as np
from utils import MulticlassMetrics
import matplotlib.pyplot as plt
import pandas as pd

class Trainer:
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
        self.optimizer = torch.optim.Adam(model.parameters, lr=self.lr)
        self.current_epoch = 0
        self.loss_per_epoch = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        if model.task == 'regression':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        else:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')

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
                                                            [dataset.target_name] == 0])
            class_1 = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                            [dataset.target_name] == 1])
            # (the remaining are nan)
            weights = torch.tensor(class_1 / class_0) if self.balance_classes else None
        else :
            patients_train = dataset.a_visit_df.patient_id.isin(dataset.train_ids)
            weights = torch.empty(self.model.num_targets, device = self.model.device)
            for w in range(self.model.num_targets):
                weights[w] = len(dataset.a_visit_df[patients_train][dataset.a_visit_df[patients_train]
                                                                    [dataset.target_name] == w])
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
        batch_valid = Batch2(model.device, partition.partitions_test[partition.current_fold],
                            partition.partitions_test[partition.current_fold], partition.partitions_test[partition.current_fold])
        batch_valid.get_batch(self.dataset, debug_patient=None)
        batch_valid.get_masks(self.dataset, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(partition.partitions_train[partition.current_fold], size=1)[0]
            #debug_patient = '0034bdeb-8b71-15f5-2006-4a49be283b3f'
            print(
                f'Debug patient {debug_patient} \nall targets \n{self.dataset.targets_df_proc[self.dataset.targets_df_proc.patient_id == debug_patient][["das283bsr_score", self.dataset.target_name]]}')

        metrics = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0, 1, 2], device = model.device))

        batch = Batch2(model.device, partition.partitions_train[partition.current_fold],
                      partition.partitions_train[partition.current_fold])
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            batch.get_batch(self.dataset, self.batch_size, debug_patient)
            batch.get_masks(
                self.dataset, debug_patient)
            self.update_epoch_and_indices(batch)
            self.loss, metrics = model.apply_and_get_loss(self.dataset, self.criterion, batch, metrics)

            # take optimizer step once loss wrt all visits has been computed
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.loss

                    model.eval()
                    metrics_val = MulticlassMetrics(device=model.device, possible_classes=torch.tensor([0,1,2], device = model.device))
                    self.loss_valid, metrics_val = model.apply_and_get_loss(self.dataset, self.criterion,
                                                                            batch_valid, metrics_val)
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid
                    metrics.get_metrics()
                    f1 = metrics.macro_f1
                    self.f1_per_epoch[self.current_epoch - 1] = f1
                    metrics_val.get_metrics()
                    f1_val = metrics_val.macro_f1
                    self.f1_per_epoch_valid[self.current_epoch - 1] = f1_val
                    print(
                        f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid} f1 {f1} f1_valid {f1_val}')
                    # re-initialize metrics for new epoch
                    metrics = MulticlassMetrics(
                        device=model.device, possible_classes=torch.tensor([0, 1, 2], device=model.device))
                    if self.use_early_stopping:
                        self.check_early_stopping()

        return f1, f1_val

    def plot_accuracy(self):
        plt.figure()
        plt.plot(range(0, len(self.accuracy_per_epoch[:self.current_epoch]),
                 1), self.accuracy_per_epoch[:self.current_epoch])
        plt.plot(range(0, len(self.accuracy_per_epoch[:self.current_epoch]), 1),
                 self.accuracy_per_epoch_valid[:self.current_epoch])
        return


class Batch2:
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
        # get max number of visits for a patient in subset
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
                    f'visit {visit} cropped timeline mask {visual} visit mask {dataset.masks.a_visit_masks[index][visit]} medication mask {dataset.masks.med_masks[index][visit]}')
    
        return


# class Batch:
#     def __init__(self, device, all_indices, available_indices, current_indices=None):
#         self.device = device
#         self.all_indices = all_indices
#         self.available_indices = available_indices
#         self.current_indices = current_indices

#     def get_batch(self, dataset, batch_size=None, debug_patient=None):
#         """ First, selects a batch of patients from the available indices for this epoch and the corresponding tensor (visits/medications/
#         patient) slices. Then for each visit v, for each patient of the batch, create a mask to combine the visits/medication events coming before v in the right order. 

#         """
#         # during training, only select subset of available indices
#         if batch_size:
#             # batch size
#             size = min(len(self.available_indices), batch_size)
#             # batch and corresponding tensor indices
#             self.current_indices = np.random.choice(self.available_indices, size=size, replace=False)
#             #print(f'len available indices {len(self.available_indices)}')

#         # get corresponding indices in tensors
#         indices_dict = {name: [] for name in dataset.event_names}
#         indices_dict['patients'] = []
#         indices_dict['targets'] = []

#         for elem in self.current_indices:
#             for name in indices_dict:
#                 df = getattr(dataset, name + '_df_proc')
#                 indices_dict[name].extend(df[df.patient_id ==
#                                              elem]['tensor_indices_train'].values)

#         for name in indices_dict:
#             setattr(self, 'indices_' + name, np.array(indices_dict[name]))

#         # for debugging
#         if debug_patient and debug_patient in self.current_indices:
#             self.debug_index = list(self.current_indices).index(debug_patient)
#             print(f'index in batch {self.debug_index}')
#         else:
#             self.debug_index = None
#         if (self.indices_a_visit != self.indices_targets).any():
#             raise ValueError('index mismatch between visits and targets')

#         return

#     def get_masks(self, dataset, debug_patient):
#         """_summary_

#         Args:
#             dataset (_type_): _description_
#             subset (_type_): _description_
#             min_num_visits (_type_): min number of initial visits to retrieve the information from
#         e.g. if min_num_visits = 2, for each patient we start retrieving all information
#         up to the 2nd visit, i.e. medications before 2nd visit and info about 1st visit
#         (in other words, min_num_visits is the first target visit). For each visit v >= min_num_visits, we store for each patient the number of visits and medication events
#         up to v

#         Returns:
#             _type_: _description_
#         """
#         # get max number of visits for a patient in subset
#         self.max_num_visits = max([len(dataset.patients[index].visits) for index in self.current_indices])
#         seq_lengths = torch.zeros(size=(self.max_num_visits - dataset.min_num_visits + 1,
#                                         len(self.current_indices), len(dataset.event_names)), dtype=torch.long, device=self.device)
#         # to store for each patient for each visit the visit/medication mask up to that visit. This mask allows
#         # us to then easily combine the visit and medication events in the right order. True is for visit events and False for medications.
#         # E.g. if a patient has the timeline [m1, m2, v1, m3, m4, v2, m5, v3] the corresponding masks up to each of the 3 visits would be
#         # [[False, False], [False, False, True, False, False], [False, False, True, False, False, True, False]] and the sequence lengths
#         # for visits/medication count up to each visit [[0, 2], [1, 4], [2, 5]]
#         masks_dict = {event: [[] for i in range(len(self.current_indices))] for event in dataset.event_names}

#         for i, patient in enumerate(self.current_indices):
#             for visit in range(0, len(dataset.patients[patient].visits) - dataset.min_num_visits + 1):
#                 # get timeline up to visit (not included)
#                 seq_lengths[visit, i, :], _, cropped_timeline_mask, visual = dataset.patients[patient].get_cropped_timeline(
#                     visit + dataset.min_num_visits)
#                 for event in dataset.event_names:
#                     # masks_dict[event][i].append(torch.broadcast_to(torch.tensor([[True if tuple_[0] == event else False] for tuple_ in cropped_timeline_mask]),
#                     #                                               (len(cropped_timeline_mask), model.size_embedding)))
#                     masks_dict[event][i].append(torch.tensor([[True if tuple_[0] == event else False] for tuple_ in cropped_timeline_mask]),
#                                                 )

#                 if debug_patient and patient == debug_patient:
#                     print(
#                         f'visit {visit} cropped timeline mask {visual} visit mask {masks_dict["a_visit"][i]} medication mask {masks_dict["med"][i]}')

#         # tensor of shape batch_size x max_num_visits with True in position (p, v) if patient p has at least v visits
#         # and False else. we use this mask later to select the patients up to each visit.
#         self.available_visit_mask = torch.tensor([[True if index <= len(dataset.patients[patient].visits)
#                                                   else False for index in range(dataset.min_num_visits, self.max_num_visits + 1)] for patient in self.current_indices], device=self.device)

#         # stores for each patient in batch the total number of visits and medications
#         # it is used later to index correctly the visits and medications dataframes
#         # total num visits and meds

#         self.total_num = torch.tensor([[getattr(dataset.patients[patient], 'num_' + event + '_events')
#                                       for event in dataset.event_names] for patient in self.current_indices], device=self.device)
#         self.seq_lengths = seq_lengths
#         for event in dataset.event_names:
#             setattr(self, event + '_masks', masks_dict[event])
#         return
