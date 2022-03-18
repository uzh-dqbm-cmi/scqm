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

def analyze_results(dataset, results_df):
    non_nan = len(results_df['predictions'].dropna())
    print(f'Number of predicitions {non_nan}')
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

    return
