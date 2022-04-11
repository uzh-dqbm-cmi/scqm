import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import pandas as pd
import math
import random
from sklearn.metrics import roc_curve, roc_auc_score

def set_seeds(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    return
        
class Results:
    def __init__(self, dataset, model):
        self.dataset = dataset
        self.model = model
    def evaluate_model(self, patient_ids):
        results_df = pd.DataFrame()
        naive_results = pd.Series()
        # for naive baseline
        patients_train = self.dataset.visits_df.patient_id.isin(self.dataset.train_ids)
        class_0 = len(self.dataset.visits_df[patients_train][self.dataset.visits_df[patients_train]
                                                        [self.dataset.target_name] == 0])
        class_1 = len(self.dataset.visits_df[patients_train][self.dataset.visits_df[patients_train]
                                                        [self.dataset.target_name] == 1])
        for patient in patient_ids:
            target_values = self.dataset[patient].targets_df[self.dataset.target_name][self.dataset.min_num_visits - 1:]
            target_categories = self.dataset[patient].targets_df[self.dataset.target_name][self.dataset.min_num_visits - 1:]
            predictions, predicted_categories = self.model.apply(self.dataset, patient)
            results_df = results_df.append(pd.DataFrame({'patient_id': patient, 'targets' : target_values, 'target_categories':target_categories, 'predicted_categories': predicted_categories, 'predictions' :torch.sigmoid(predictions).flatten().detach()}))
            #for naive baseline
            # in that case, naive baseline is no change in das28 (i.e. same category as at last visit)
            if self.dataset.target_name == 'das28_category':
                naive_categories = pd.Series(self.dataset[patient].targets_df['das28_category'][self.dataset.min_num_visits-2:-1])
            else :
                # in that case, naive baseline is 0 with proba #class0_in_train/length(train) and class 1 with proba #class1_in_train/length(train)
                naive_categories = pd.Series(np.random.binomial(1, class_1/(class_0 + class_1), size = len(target_categories)))
                # other baseline : first predicition is random, and then predict value of previous target (i.e. predict increasing if it was before)
                # naive_results = naive_results.append(pd.Series(np.random.binomial(
                #     1, class_1 / (class_0 + class_1), size=1)))
                # naive_categories = pd.Series(
                #     self.dataset[patient].targets_df[self.dataset.target_name][self.dataset.min_num_visits - 1:-1])
            naive_results = naive_results.append(naive_categories)
        metrics = Metrics(self.model.device, results_df['predicted_categories'], results_df['target_categories'], results_df['predictions'])
        metrics_naive_baseline = Metrics(self.model.device, naive_results, results_df['target_categories'])
        return results_df, metrics, metrics_naive_baseline


class Metrics:
    def __init__(self, device, predictions=None, true_values=None):
        if predictions is None:
            self.predictions = torch.empty(0, device=device)
            self.true_values = torch.empty(0, device=device)

        elif isinstance(predictions, pd.Series):
            self.predictions = torch.tensor(predictions.values)
            self.true_values = torch.tensor(true_values.values)

        else:
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
        return 1 / len(self) * torch.sum((self.predictions - self.true_values)**2)

class BinaryMetrics(Metrics):
    def __init__(self, device, predictions = None, true_values=None, predicted_probas = None):
        super().__init__(device, predictions, true_values)
        if predictions is None :
            self.predicted_probas = torch.empty(0, device=device)
        elif isinstance(predictions, pd.Series):
            self.predicted_probas = torch.tensor(
                predicted_probas.values) if predicted_probas is not None else self.predictions
        else : 
            self.predicted_probas = predicted_probas if predicted_probas else predictions
    def discrete_metrics(self, print_confusion =False):
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
        if print_confusion:
            print(pd.DataFrame(data = [[self.TN, self.FN], [self.FP, self.TP]], index = ['pred 0', 'pred 1'], columns = ['true 0', 'true 1']))
    def get_auroc(self):
        fpr, tpr, thresholds = roc_curve(self.true_values, self.predicted_probas, pos_label=1)
        auc = roc_auc_score(self.true_values, self.predicted_probas)
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label=f"ROC curve (AUC = {np.round(auc,2)})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        plt.show()


class MulticlassMetrics(Metrics):
    def __init__(self, device, possible_classes, predictions=None, true_values=None, predicted_probas=None):
        super().__init__(device, predictions, true_values)
        self.possible_classes = possible_classes
        if predictions is None:
            self.predicted_probas = torch.empty(0, device=device)
        elif isinstance(predictions, pd.Series):
            self.predicted_probas = torch.tensor(
                predicted_probas.values) if predicted_probas is not None else self.predictions
        else:
            self.predicted_probas = predicted_probas if predicted_probas else predictions
    def get_metrics(self):
        self.macro_f1 = 0
        self.fpr = np.empty(len(self.possible_classes))
        self.tpr = np.empty(len(self.possible_classes))
        for class_ in self.possible_classes:
            TP = len([elem for index, elem in enumerate(self.predictions)
                     if elem == class_ and elem == self.true_values[index]])
            FP = len([elem for index, elem in enumerate(self.predictions)
                        if elem == class_ and elem != self.true_values[index]])
            FN = len([elem for index, elem in enumerate(self.predictions)
                      if elem != class_ and class_ == self.true_values[index]])
            TN = len([elem for index, elem in enumerate(self.predictions)
                      if elem != class_ and class_ != self.true_values[index]])
            if TP + FN != len([elem for elem in self.true_values if elem == class_]):
                raise(ArithmeticError('number of positives dont match'))
            if TP + FP == 0:
                precision= 0
            else:
                precision = TP/(TP + FP)
            if TP + FN == 0:
                recall=0
            else:
                recall = TP/(TP+FN)
            if precision + recall == 0:
                print(f'Setting f1 to zero for class {class_} because division by zero')
                F1 =0
            else:
                F1 = 2*(precision*recall)/(precision + recall)
            self.macro_f1 += F1
            self.fpr[class_] = FP/(FP+TN)
            self.tpr[class_] = TP/(TP+FN)
        self.macro_f1 = self.macro_f1/len(self.possible_classes)        
        return
    def get_auroc(self):
        true_values_one_hot = F.one_hot(self.true_values)
        plt.figure()
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.legend(loc="lower right")
        for class_ in self.possible_classes:
            fpr, tpr, thresholds = roc_curve(true_values_one_hot[:,class_], self.predicted_probas[:, class_], pos_label=1)
            #auc = roc_auc_score(self.true_values, self.predicted_probas)
            lw = 2
            plt.plot(
                fpr,
                tpr,
                lw=lw,
                label=f"ROC curve class {class_}",
            )
        plt.show()
       


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


class Masks:
    def __init__(self, device, indices):
        self.device = device
        self.indices = indices

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
        self.num_visits = [len(dataset.patients[index].visits) for index in self.indices]
        max_num_visits = max(self.num_visits)
        seq_lengths = torch.zeros(size=(max_num_visits - dataset.min_num_visits + 1,
                                        len(self.indices), len(dataset.event_names)), dtype=torch.long, device=self.device)
        # to store for each patient for each visit the visit/medication mask up to that visit. This mask allows
        # us to then easily combine the visit and medication events in the right order. True is for visit events and False for medications.
        # E.g. if a patient has the timeline [m1, m2, v1, m3, m4, v2, m5, v3] the corresponding masks up to each of the 3 visits would be
        # [[False, False], [False, False, True, False, False], [False, False, True, False, False, True, False]] and the sequence lengths
        # for visits/medication count up to each visit [[0, 2], [1, 4], [2, 5]]
        masks_dict = {event: [[] for i in range(len(self.indices))] for event in dataset.event_names}

        for i, patient in enumerate(self.indices):
            for visit in range(0, len(dataset.patients[patient].visits) - dataset.min_num_visits + 1):
                # get timeline up to visit (not included)
                seq_lengths[visit, i, :], _, cropped_timeline_mask, visual = dataset.patients[patient].get_cropped_timeline(
                    visit + dataset.min_num_visits)
                for event in dataset.event_names:
                    # masks_dict[event][i].append(torch.broadcast_to(torch.tensor([[True if tuple_[0] == event else False] for tuple_ in cropped_timeline_mask]),
                    #                                               (len(cropped_timeline_mask), model.size_embedding)))
                    masks_dict[event][i].append(torch.tensor([[True if tuple_[0] == event else False] for tuple_ in cropped_timeline_mask]),
                                                )

                if debug_patient and patient == debug_patient:
                    print(
                        f'visit {visit} cropped timeline mask {visual} visit mask {masks_dict["a_visit"][i]} medication mask {masks_dict["med"][i]}')

        # tensor of shape batch_size x max_num_visits with True in position (p, v) if patient p has at least v visits
        # and False else. we use this mask later to select the patients up to each visit.
        self.available_visit_mask = torch.tensor([[True if index <= len(dataset.patients[patient].visits)
                                                  else False for index in range(dataset.min_num_visits, max_num_visits + 1)] for patient in self.indices], device=self.device)

        # stores for each patient in batch the total number of visits and medications
        # it is used later to index correctly the visits and medications dataframes
        # total num visits and meds

        self.total_num = torch.tensor([[getattr(dataset.patients[patient], 'num_' + event + '_events')
                                      for event in dataset.event_names] for patient in self.indices], device=self.device)
        self.seq_lengths = seq_lengths
        for event in dataset.event_names:
            setattr(self, event + '_masks', masks_dict[event])
        return


