import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import math



class DataPartition:
    def __init__(self, dataset, k):
        self.dataset = dataset
        self.k = k
        self.split()
        
    def split(self):
        # split data into train and test (no valid)
        self.dataset.split_data(prop_valid=0.0, prop_test=0.2)
        self.dataset.scale_and_tensor()
        # get partition of size k of train set
        self.fold_size = math.ceil(len(self.dataset.train_ids)/self.k)
        self.permuted_ids = np.random.permutation(self.dataset.train_ids) 
        self.partitions_test = {test_fold: self.permuted_ids[self.fold_size *
                                                        test_fold: self.fold_size * (1 + test_fold)] for test_fold in range(self.k)}
        self.partitions_train = {train_fold: np.array(
            [id_ for id_ in self.permuted_ids if id_ not in self.partitions_test[train_fold]]) for train_fold in range(self.k)}

    def get_tensor(self, df='visits_df', subset = 'train', fold = 0):
        # e.g. dataset.visits_df_scaled_tensor_train[dataset.visits_df_proc[dataset.visits_df_proc.patient_id.isin(partition.partitions[0])].tensor_indices_train.values]
        tensor_name = df + '_scaled_tensor_train'
        tensor = getattr(self.dataset, tensor_name)
        processed_df = getattr(self.dataset, df + '_proc')
        if subset == 'test':
            tensor_slice = tensor[processed_df[processed_df.patient_id.isin(self.partitions_test[fold])].tensor_indices_train.values]
        else :
            tensor_slice = tensor[processed_df[processed_df.patient_id.isin(
                self.partitions_train[fold])].tensor_indices_train.values]
        return tensor_slice
    
    def set_current_fold(self, k):
        self.current_fold = k

class Trainer:
    def __init__(self, model, dataset, n_epochs, batch_size, lr, balance_classes):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.balance_classes = balance_classes
        self.early_stopping = False
        self.optimizer = torch.optim.Adam(model.parameters, lr=self.lr)
        self.current_epoch = 0
        self.loss_per_epoch = torch.empty(size=(n_epochs, 1))
        self.loss_per_epoch_valid = torch.empty(size=(n_epochs, 1))
        if model.task == 'regression':
            self.criterion = torch.nn.MSELoss(reduction='sum')
        else :
            self.pos_weight = self.get_target_weighting(dataset)
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction = 'sum', pos_weight=self.pos_weight)
    
    def get_target_weighting(self, dataset):
        patients_train = dataset.visits_df.patient_id.isin(dataset.train_ids)
        class_0 = len(dataset.visits_df[patients_train][dataset.visits_df[patients_train]
                                                        ['das28_category'] == 0])
        class_1 = len(dataset.visits_df[patients_train][dataset.visits_df[patients_train]
                                                        ['das28_category'] == 1])
        pos_weight = torch.tensor(class_1 / class_0) if self.balance_classes else None
        return pos_weight
    
    def check_early_stopping(self):
        if self.current_epoch > 35 and (self.loss_per_epoch_valid[self.current_epoch - 1 - 20:self.current_epoch - 1] / self.loss_per_epoch_valid[self.current_epoch - 1 - 30:self.current_epoch - 1 - 10]).mean() > 1:
            self.early_stopping = True
            self.loss_per_epoch = self.loss[:self.current_epoch-1]
            self.loss_per_epoch_valid = self.loss_per_epoch_valid[:self.current_epoch-1]
            print('early stopping because validation loss is increasing')

    def train_model(self, model, dataset, partition, debug_patient):

        print(f'device {model.device}')
        #dfs and tensors
        dataset.move_to_device(model.device)

        # validation
        # debug_patient_valid = np.random.choice(partition.partitions_test[partition.current_fold], size=1)[0]
        # debug_index_valid = list(partition.partitions_test[partition.current_fold]).index(debug_patient_valid)
        # print(
        #     f'Debug patient {debug_patient_valid} \nall targets \n{dataset.targets_df_proc[dataset.targets_df_proc.patient_id == debug_patient_valid]["das283bsr_score"]}')
        batch_valid = Batch(partition.partitions_test[partition.current_fold],
                            partition.partitions_test[partition.current_fold], partition.partitions_test[partition.current_fold], debug_index=None)
        batch_valid.get_indices_valid(partition, dataset)
        batch_valid.get_masks(dataset, model, debug_patient=None)

        # debug patient
        if debug_patient:
            debug_patient = np.random.choice(partition.partitions_train[partition.current_fold], size=1)[0]
            print(
                f'Debug patient {debug_patient} \nall targets \n{dataset.targets_df_proc[dataset.targets_df_proc.patient_id == debug_patient]["das283bsr_score"]}')

        metrics = Metrics(device=model.device)

        batch = Batch(partition.partitions_train[partition.current_fold],
                    partition.partitions_train[partition.current_fold])
        while (self.current_epoch < self.n_epochs) and self.early_stopping == False:
            model.train()
            # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
            # patients with a given number of visits

            batch.get_batch_and_masks(self, dataset, model, debug_patient)

            self.loss, metrics = model.apply_and_get_loss(dataset, self.criterion, batch, metrics)

            # take optimizer step once loss wrt all visits has been computed
            self.optimizer.zero_grad()
            self.loss.backward()
            self.optimizer.step()

            # store loss and evaluate on validation data
            if len(batch.available_indices) == len(batch.all_indices):
                with torch.no_grad():
                    self.loss_per_epoch[self.current_epoch - 1] = self.loss

                    model.eval()
                    metrics_val = Metrics(device=model.device)
                    self.loss_valid, metrics_val = model.apply_and_get_loss(dataset, self.criterion,
                                                                            batch_valid, metrics_val)
                    self.loss_per_epoch_valid[self.current_epoch - 1] = self.loss_valid
                    metrics.discrete_metrics()
                    accuracy = metrics.accuracy
                    metrics_val.discrete_metrics()
                    accuracy_val = metrics_val.accuracy
                    print(
                        f'epoch : {self.current_epoch} loss {self.loss} loss_valid {self.loss_valid} accuracy {accuracy} accuracy_valid {accuracy_val}')
                    # re-initialize metrics for new epoch
                    metrics = Metrics(device=model.device)
                    self.check_early_stopping()

        return accuracy, accuracy_val

class Batch:
    def __init__(self, all_indices, available_indices, current_indices = None, debug_index = None):
        self.all_indices = all_indices
        self.available_indices = available_indices
        self.current_indices = current_indices
        self.debug_index = debug_index

    def get_batch_and_masks(self, trainer, dataset, model, debug_patient=None):
        """ First, selects a batch of patients from the available indices for this epoch and the corresponding tensor (visits/medications/
        patient) slices. Then for each visit v, for each patient of the batch, create a mask to combine the visits/medication events coming before v in the right order. 

        Args:
            epoch (_type_): current epoch
            indices (_type_): available indices to select batch from
            dataset (_type_): dataset object
            batch_size (_type_): batch size
            df_v (_type_): visits dataframe
            df_m (_type_): medications dataframe
            df_p (_type_): patients dataframe
            tensor_v (_type_): visits tensor
            tensor_m (_type_): medications tensor
            tensor_p (_type_): patients tensor

        Returns: t_v, t_m, t_p, seq_lengths, masks, indices, epoch, max_num_visits, visit_mask, total_num_visits_and_meds
            _type_:
        """
        # batch size
        size = min(len(self.available_indices), trainer.batch_size)
        # batch and corresponding tensor indices
        self.current_indices = np.random.choice(self.available_indices, size=size, replace=False)

        # get corresponding indices in tensors

        tensor_indices_visits, tensor_indices_pat, tensor_indices_meds, tensor_indices_targ = [], [], [], []

        for elem in self.current_indices:
            tensor_indices_visits.extend(
                dataset.visits_df_proc[dataset.visits_df_proc.patient_id == elem]['tensor_indices_train'].values)
            tensor_indices_meds.extend(
                dataset.medications_df_proc[dataset.medications_df_proc.patient_id == elem]['tensor_indices_train'].values)
            tensor_indices_pat.extend(
                dataset.patients_df_proc[dataset.patients_df_proc.patient_id == elem]['tensor_indices_train'].values)
            tensor_indices_targ.extend(
                dataset.targets_df_proc[dataset.targets_df_proc.patient_id == elem]['tensor_indices_train'].values)
        self.indices_v = np.array(tensor_indices_visits)
        self.indices_m = np.array(tensor_indices_meds)
        self.indices_p = np.array(tensor_indices_pat)
        self.indices_t = np.array(tensor_indices_targ)
        # for debugging
        if debug_patient and debug_patient in self.current_indices:
            self.debug_index = list(self.current_indices).index(debug_patient)
            print(f'index in batch {self.debug_index}')
        else :
            self.debug_index = None
        if (self.indices_v != self.indices_t).any():
            raise ValueError('index mismatch between visits and targets')

        self.get_masks(
            dataset, model, debug_patient)
        # remove batch indices from available indices (since one epoch is one pass through whole data set)
        self.available_indices = np.array([x for x in self.available_indices if x not in self.current_indices])
        # a whole pass through the data has been completed
        if len(self.available_indices) == 0:
            self.available_indices = self.all_indices
            trainer.current_epoch += 1

        return 

    def get_masks(self, dataset, model, debug_patient):
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
        self.max_num_visits = max([len(dataset.patients[index].visits) for index in self.current_indices])
        seq_lengths = torch.zeros(size=(self.max_num_visits - dataset.min_num_visits + 1,
                                        len(self.current_indices), 2), dtype=torch.long)
        # to store for each patient for each visit the visit/medication mask up to that visit. This mask allows
        # us to then easily combine the visit and medication events in the right order. True is for visit events and False for medications.
        # E.g. if a patient has the timeline [m1, m2, v1, m3, m4, v2, m5, v3] the corresponding masks up to each of the 3 visits would be
        # [[False, False], [False, False, True, False, False], [False, False, True, False, False, True, False]] and the sequence lengths
        # for visits/medication count up to each visit [[0, 2], [1, 4], [2, 5]]
        masks = [[] for i in range(len(self.current_indices))]
        for i, patient in enumerate(self.current_indices):
            for visit in range(0, len(dataset.patients[patient].visits) - dataset.min_num_visits + 1):
                # get timeline up to visit (not included)
                seq_lengths[visit, i, 0], seq_lengths[visit, i,
                                                    1], _, cropped_timeline_mask, visual = dataset.patients[patient].get_cropped_timeline(visit + dataset.min_num_visits)
                masks[i].append(torch.broadcast_to(torch.tensor([[tuple_[0]] for tuple_ in cropped_timeline_mask]),
                                                (len(cropped_timeline_mask), model.VEncoder.size_embedding)))
                if debug_patient and patient == debug_patient:
                    print(f'visit {visit} cropped timeline mask {visual} ')

        # tensor of shape batch_size x max_num_visits with True in position (p, v) if patient p has at least v visits
        # and False else. we use this mask later to select the patients up to each visit.
        self.visit_mask = torch.tensor([[True if index <= len(dataset.patients[patient].visits)
                                    else False for index in range(dataset.min_num_visits, self.max_num_visits + 1)] for patient in self.current_indices])

        # stores for each patient in batch the total number of visits and medications
        # it is used later to index correctly the visits and medications dataframes
        #total num visits and meds
        self.total_num = torch.tensor(
            [[len(dataset.patients[patient].visits), dataset.patients[patient].num_med_events] for patient in self.current_indices])
        
        self.seq_lengths = seq_lengths
        self.masks = masks
        return 
    
    def get_indices_valid(self, partition, dataset):
        tensor_indices_visits_val, tensor_indices_pat_val, tensor_indices_meds_val, tensor_indices_targ_val = [], [], [], []

        for elem in partition.partitions_test[partition.current_fold]:
            tensor_indices_visits_val.extend(
                dataset.visits_df_proc[dataset.visits_df_proc.patient_id == elem]['tensor_indices_train'].values)
            tensor_indices_meds_val.extend(
                dataset.medications_df_proc[dataset.medications_df_proc.patient_id == elem]['tensor_indices_train'].values)
            tensor_indices_pat_val.extend(
                dataset.patients_df_proc[dataset.patients_df_proc.patient_id == elem]['tensor_indices_train'].values)
            tensor_indices_targ_val.extend(
                dataset.targets_df_proc[dataset.targets_df_proc.patient_id == elem]['tensor_indices_train'].values)

        self.indices_v = np.array(tensor_indices_visits_val)
        self.indices_m = np.array(tensor_indices_meds_val)
        self.indices_p = np.array(tensor_indices_pat_val)
        self.indices_t = np.array(tensor_indices_targ_val)

class Metrics:
    def __init__(self, device, predictions = None, true_values=None):
        if predictions is None :
            self.predictions = torch.empty(0, device = device)
            self.true_values = torch.empty(0, device = device)
        else : 
            self.predictions = predictions
            self.true_values = true_values

    def __len__(self):
        if len(self.predictions) != len(self.true_values):    
            raise ValueError('length of predictions doesnt match true values')
        else:
            return len(self.predictions)
    def add_observations(self, new_predictions, new_true_values):
        self.predictions = torch.cat([self.predictions, new_predictions])
        self.true_values = torch.cat([self.true_values, new_true_values])
        return
    def mse(self):
        return 1/len(self)* torch.sum((self.predictions - self.true_values)**2)
    def discrete_metrics(self):
        self.TP = len([elem for index, elem in enumerate(self.predictions) if elem == 1 and elem == self.true_values[index]])
        self.TN = len([elem for index, elem in enumerate(self.predictions)
                      if elem == 0 and elem == self.true_values[index]])
        self.FP = len([elem for index, elem in enumerate(self.predictions)
                      if elem == 1 and elem != self.true_values[index]])
        self.FN = len([elem for index, elem in enumerate(self.predictions)
                      if elem == 0 and elem != self.true_values[index]])
        if self.TP + self.TN + self.FP + self.FN != len(self):
            raise ArithmeticError('Sum of TP, TN, FP, FN doesnt match len of self')
        self.sensitivity = self.TP/(self.TP + self.FN)
        self.specificity = self.TN/(self.TN + self.FP)
        self.accuracy = (self.TP + self.TN)/(self.TP + self.TN + self.FP + self.FN)
        self.f1 = self.TP/(self.TP + 1/2 * (self.FP + self.FN))
       


def create_results_df(device, subset, dataset, results, algo='adanet', num_targets=None):
    # join the dataframes with unormalized values, normalized values and predicted values
    tmp_1 = dataset.targets_df[dataset.targets_df.patient_id.isin(subset)].copy()
    tmp_2 = dataset.targets_df_proc[dataset.targets_df_proc.patient_id.isin(
        subset)][['patient_id', 'visit_date', 'das283bsr_score']]
    tmp_3 = pd.concat([tmp_1, tmp_2.rename(
        columns={'visit_date': 'visit_date_scaled', 'das283bsr_score': 'das28_scaled'})], axis=1)
    results_df = tmp_3.loc[:, ~tmp_3.columns.duplicated()]
    results = results.cpu()
    # different results shape
    if algo == 'adanet':
        for index, elem in enumerate(results_df['patient_id'].unique()):
            predictions = [np.nan] + [value for value in results[index, :] if value != 100.0]
            results_df.loc[results_df.patient_id == elem, 'scaled_predictions'] = predictions

    else:
        index_in_results = 0
        for index, elem in enumerate(results_df['patient_id'].unique()):

            predictions = torch.cat(
                [torch.tensor([np.nan]), results[index_in_results:index_in_results + int(num_targets[index])].flatten()])

            results_df.loc[results_df.patient_id == elem, 'scaled_predictions'] = np.array(predictions)
            index_in_results += int(num_targets[index])

    results_df['predictions'] = results_df['scaled_predictions'] * (dataset.visits_df_scaling_values[1]['das283bsr_score'] -
                                                                    dataset.visits_df_scaling_values[0]['das283bsr_score']) + dataset.visits_df_scaling_values[0]['das283bsr_score']
    results_df['squarred_error'] = (results_df['predictions'] - results_df['das283bsr_score'])**2
    results_df['history_length'] = [index for patient in results_df['patient_id'].unique()
                                    for index in range(len(results_df[results_df['patient_id'] == patient]))]
    results_df['days_to_prev_visit'] = np.nan
    results_df['days_to_prev_visit'][1:] = (
        results_df['visit_date'].values[1:] - results_df['visit_date'].values[:-1]).astype('timedelta64[D]') / np.timedelta64(1, 'D')
    results_df['days_to_prev_visit'] = [elem if results_df.iloc[index]['history_length']
                                        != 0 else np.nan for index, elem in enumerate(results_df['days_to_prev_visit'])]
    results_df['squarred_error_naive_baseline'] = np.nan
    results_df['squarred_error_naive_baseline'][1:] = (
        results_df['das283bsr_score'].values[1:] - results_df['das283bsr_score'].values[:-1])**2
    results_df['squarred_error_naive_baseline'] = [elem if results_df.iloc[index]['history_length']
                                                   != 0 else np.nan for index, elem in enumerate(results_df['squarred_error_naive_baseline'])]

    return results_df

def analyze_results(dataset, results_df, task = 'regression'):
    non_nan = len(results_df['predictions'].dropna())
    print(f'Number of predicitions {non_nan}')
    if task == 'regression':
        print(f'MSE between targets and prediction {results_df["squarred_error"].sum()/non_nan}')
        print(f'MSE between prediction and previous visit value'
            f' {np.nansum(((results_df["predictions"][1:].values - results_df["das283bsr_score"][:-1].values)**2))/non_nan}')
        print(
            f'MSE between das28 and das28 at previous visit (naive baseline) {results_df["squarred_error_naive_baseline"].sum()/non_nan}')
        # f1 = plt.figure()
        # plt.scatter(results_df['days_to_prev_visit'], results_df['squarred_error'], marker='x', alpha=0.5)
        # plt.xlabel('days to previous visita')
        # plt.ylabel('Squarred error')
        # plt.xlim(0,1000)
        f2 = plt.figure()
        plt.scatter(results_df['history_length'], results_df['squarred_error'], marker='x', alpha=0.5)

        plt.xlabel('history length')
        plt.ylabel('Squarred error')
    else :
        reduced = results_df[results_df['scaled_predictions'].notna()]
        conf_matrix = pd.crosstab(reduced['scaled_predictions'], reduced['das28_category'])
        print(conf_matrix)
        print(f'accuracy : {(conf_matrix.loc[0, 0] + conf_matrix.loc[1, 1])/conf_matrix.sum().sum()}')
        print(f'sensitivity : {(conf_matrix.loc[1, 1])/(conf_matrix.loc[1,1] + conf_matrix.loc[0, 1])}')
        print(f'specificity : {conf_matrix.loc[0, 0]/(conf_matrix.loc[0, 0] + conf_matrix.loc[1,0])}')
    return

def compute_metrics(device, output, targets):
    predictions = torch.tensor([1 if elem >= 0.5 else 0 for elem in output], device = device)
    TP = len([elem for index, elem in enumerate(predictions) if elem==targets[index] and elem == 1])
    TN = len([elem for index, elem in enumerate(predictions) if elem == targets[index] and elem == 0])
    FP = len([elem for index, elem in enumerate(predictions) if elem != targets[index] and elem == 1])
    FN = len([elem for index, elem in enumerate(predictions) if elem != targets[index] and elem == 0])
    
    return (TP, TN, FP, FN)
