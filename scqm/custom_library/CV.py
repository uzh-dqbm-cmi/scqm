from scqm.custom_library.models import Adaptivenet
from scqm.custom_library.training import AdaptivenetTrainer
from scqm.custom_library.partition import DataPartition
from scqm.custom_library.preprocessing import load_dfs, preprocessing, extract_adanet_features
from scqm.custom_library.data_objects import Dataset
import itertools
import numpy as np
import sys
import csv
import random
import time
import torch

from scqm.custom_library.utils import set_seeds


#TODO early stopping as parameter
#TODO shell script for cv
#TODO cleaner preprocessing
#TODO testing module
#TODO modularise baseline
#TODO stratifier
#TODO prediction module with 4 classes instead of 2

class CVWrapper:
    def __init__(self, dataset, k):
        set_seeds(0)
        self.dataset = dataset
        self.k = k
        self.partition = DataPartition(self.dataset, k=self.k)

    def set_grid(self, parameters: dict):
        self.parameter_names = list(parameters.keys())
        self.parameters = parameters

    def perform_cv(self):
        raise NotImplementedError

class CVAdaptivenet(CVWrapper):
    def perform_cv(self, fold, n_epochs = 400, file='/cluster/home/ctrottet/code/scqm_cv_results/', search='random', num_combi=40):
        filename = file + time.strftime("%Y%m%d-%H%M") + '_fold_' + str(fold) + '.csv'
        with open(filename, 'w') as f:
            header = self.parameter_names + ['epochs', 'loss', 'loss_valid',
                    'accuracy', 'accuracy_valid']
            writer = csv.writer(f)
            writer.writerow(header)
            combinations = list(itertools.product(*self.parameters.values()))
            self.partition.set_current_fold(fold)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            task = 'classification'
            # instantiate model
            num_visit_features = self.dataset.visits_df_scaled_tensor_train.shape[1]
            num_medications_features = self.dataset.medications_df_scaled_tensor_train.shape[1]
            num_general_features = self.dataset.patients_df_scaled_tensor_train.shape[1]
            batch_first = True
            # random search instead of grid search
            if search == 'random':
                combinations = random.sample(combinations, num_combi)
            for ind, (size_embedding, num_layers_enc, hidden_enc, size_history, num_layers, num_layers_pred, hidden_pred, lr, p, bal) in enumerate(combinations):
                print(f'{ind} combination out of {len(combinations)}')
                print(
                    f'size_embedding {size_embedding}, num_layers_enc {num_layers_enc}, hidden_enc {hidden_enc}, size_history {size_history}, num_layers {num_layers}, num_layers_pred {num_layers_pred}, hidden_pred {hidden_pred}, dropout {p}, lr {lr}')
                model_specifics = {'size_embedding': size_embedding, 'num_layers_enc': num_layers_enc, 'hidden_enc': hidden_enc, 'size_history': size_history, 'num_layers': num_layers, 'num_layers_pred': num_layers_pred, 'hidden_pred': hidden_pred,
                                'num_visit_features': num_visit_features, 'num_medications_features': num_medications_features, 'num_general_features': num_general_features, 'dropout': p, 'batch_first': batch_first, 'device': device, 'task': task}
                model = Adaptivenet(model_specifics, device)
                self.dataset.min_num_visits = 2
                trainer = AdaptivenetTrainer(model, self.dataset, n_epochs, batch_size=32, lr=lr, balance_classes=bal, use_early_stopping = True)
                accuracy, accuracy_valid = trainer.train_model(model, self.partition, debug_patient=False)
                writer.writerow(np.array([size_embedding, num_layers_enc, hidden_enc, size_history,
                                num_layers, num_layers_pred, hidden_pred, lr, p, bal, trainer.current_epoch, trainer.loss.item(), trainer.loss_valid.item(), accuracy, accuracy_valid]))


if __name__ == '__main__':

    df_dict = load_dfs()
    df_dict = preprocessing(df_dict)
    patients_df, medications_df, visits_df, targets_df, _ = extract_adanet_features(df_dict, das28=True)
    df_dict_anet = {'visits': visits_df, 'patients': patients_df, 'medications': medications_df, 'targets': targets_df}
    dataset = Dataset(df_dict_anet, df_dict_anet['patients']['patient_id'].unique(), target_name = 'das28_increase')
    # keep only patients with more than two visits
    dataset.drop([id_ for id_, patient in dataset.patients.items() if len(patient.visit_ids) <= 2])
    print(f'Dropping patients with less than 3 visits, keeping {len(dataset)}')
    # prepare for training
    dataset.transform_to_numeric_adanet()

    cv = CVAdaptivenet(dataset, k=5)
    parameters = {
    "SIZE_EMBEDDING" : np.array([3, 5, 10]),
    "NUM_LAYERS_ENC" : np.array([1, 2, 5, 10]),
    "HIDDEN_ENC" : np.array([20, 50, 100]),
    "SIZE_HISTORY" : np.array([3, 5, 10]),
    "NUM_LAYERS" : np.array([1, 2, 5]),
    "NUM_LAYERS_PRED" : np.array([1, 2, 5]),
    "HIDDEN_PRED" : np.array([20, 50, 100]),
    "LR" : np.array([1e-3]),
    "P" : np.array([0.0, 0.1, 0.2]),
    "BALANCE_CLASSES" : np.array([False, True])}
    cv.set_grid(parameters)
    fold = int(sys.argv[1])
    print(f'fold {fold}')
    cv.perform_cv(fold=fold, n_epochs=400)

