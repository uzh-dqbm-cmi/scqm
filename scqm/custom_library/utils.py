import numpy as np
import matplotlib.pyplot as plt
import torch
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
    def __init__(self, device, predictions = None, true_values=None, predicted_probas = None):
        if predictions is None :
            self.predictions = torch.empty(0, device = device)
            self.true_values = torch.empty(0, device = device)
            self.predicted_probas = torch.empty(0, device=device)
        elif isinstance(predictions, pd.Series):
            self.predictions = torch.tensor(predictions.values)
            self.true_values = torch.tensor(true_values.values)
            self.predicted_probas = torch.tensor(
                predicted_probas.values) if predicted_probas is not None else self.predictions

        else : 
            self.predictions = predictions
            self.true_values = true_values
            self.predicted_probas = predicted_probas if predicted_probas else predictions

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

