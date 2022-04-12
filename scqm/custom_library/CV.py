from models import Adaptivenet
from training import AdaptivenetTrainer
from partition import DataPartition
from preprocessing import load_dfs, load_dfs_all_data, preprocessing, extract_adanet_features
from data_objects import Dataset
import itertools
import numpy as np
import sys
import csv
import random
import time
import torch
import pickle

from utils import set_seeds


#TODO early stopping as parameter
#TODO shell script for cv
#TODO cleaner preprocessing
#TODO testing module
#TODO modularise baseline
#TODO stratifier
#TODO decoders in AE style
#TODO find out which tables/features are time dependent and which are general. Implement separate encoders for each "event" i.e. dataframe and try one for several/all
#TODO add prediction of das28 as stable (delta <= 1.2)

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
    def perform_cv(self, fold, n_epochs = 400, file='/cluster/home/ctrottet/code/scqm_cv_results/', search='random', num_combi=1):
        filename = file + time.strftime("%Y%m%d-%H%M") + '_fold_' + str(fold) + '.csv'
        with open(filename, 'w') as f:
            header = self.parameter_names + ['epochs', 'loss', 'loss_valid',
                    'f1', 'f1_valid']
            writer = csv.writer(f)
            writer.writerow(header)
            combinations = list(itertools.product(*self.parameters.values()))
            self.partition.set_current_fold(fold)
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            task = 'classification'

            # instantiate model
            num_visit_features = self.dataset.a_visit_df_scaled_tensor_train.shape[1]
            num_medications_features = self.dataset.med_df_scaled_tensor_train.shape[1]
            num_general_features = self.dataset.patients_df_scaled_tensor_train.shape[1]
            num_socio_features = self.dataset.socio_df_scaled_tensor_train.shape[1]
            num_radai_features = self.dataset.radai_df_scaled_tensor_train.shape[1]
            num_haq_features = self.dataset.haq_df_scaled_tensor_train.shape[1]
            batch_first = True
            # random search instead of grid search
            if search == 'random':
                combinations = random.sample(combinations, num_combi)
            for ind, (size_embedding, num_layers_enc, hidden_enc, size_history, num_layers, num_layers_pred, hidden_pred, lr, p, bal) in enumerate(combinations):
                print(f'{ind} combination out of {len(combinations)}')
                print(
                    f'size_embedding {size_embedding}, num_layers_enc {num_layers_enc}, hidden_enc {hidden_enc}, size_history {size_history}, num_layers {num_layers}, num_layers_pred {num_layers_pred}, hidden_pred {hidden_pred}, dropout {p}, lr {lr}')
                model_specifics = {'num_targets':3,'size_embedding': size_embedding, 'num_layers_enc': num_layers_enc, 'hidden_enc': hidden_enc, 'size_history': size_history, 'num_layers': num_layers, 'num_layers_pred': num_layers_pred, 'hidden_pred': hidden_pred,
                                   'event_names': ['a_visit', 'med', 'socio', 'radai', 'haq'], 'a_visit' : {'num_features': num_visit_features}, 'med' : {'num_features': num_medications_features},
                'socio': {'num_features': num_socio_features}, 'radai': {'num_features': num_radai_features}, 'haq': {'num_features': num_haq_features}, 'num_general_features': num_general_features, 'dropout': p, 'batch_first': batch_first, 'device': device, 'task': task}
                model = Adaptivenet(model_specifics, device)
                self.dataset.min_num_visits = 2
                trainer = AdaptivenetTrainer(model, self.dataset, n_epochs, batch_size=int(len(dataset)/15), lr=lr, balance_classes=bal, use_early_stopping = True)
                f1, f1_valid = trainer.train_model(model, self.partition, debug_patient=False)
                writer.writerow(np.array([size_embedding, num_layers_enc, hidden_enc, size_history,
                                num_layers, num_layers_pred, hidden_pred, lr, p, bal, trainer.current_epoch, trainer.loss.item(), trainer.loss_valid.item(), f1, f1_valid]))


if __name__ == '__main__':

    # df_dict = load_dfs_all_data()
    # df_dict = preprocessing(df_dict)
    # patients_df, medications_df, visits_df, targets_df, socioeco_df, radai_df, haq_df, _ = extract_adanet_features(
    #     df_dict, only_meds=True, das28=True)
    # df_dict_anet = {'a_visit': visits_df, 'patients': patients_df, 'med': medications_df, 'targets': targets_df,
    #                 'socio': socioeco_df, 'radai': radai_df, 'haq': haq_df}
    # dataset = Dataset(df_dict_anet, df_dict_anet['patients']['patient_id'].unique(
    # ), 'das28_increase', ['a_visit', 'med', 'socio', 'radai', 'haq'])
    with open('/opt/data/processed/saved_dataset.pickle', 'rb') as handle:
        dataset = pickle.load(handle)
    patients_to_drop = np.random.choice(dataset.patient_ids, size=int(len(dataset)/2), replace=False)
    dataset.drop(patients_to_drop)
    # prepare for training
    dataset.create_dfs()
    dataset.transform_to_numeric_adanet()

    cv = CVAdaptivenet(dataset, k=5)
    parameters = {
    "SIZE_EMBEDDING" : np.array([3, 5, 10]),
    "NUM_LAYERS_ENC" : np.array([1, 5, 10, 15]),
    "HIDDEN_ENC" : np.array([50, 100]),
    "SIZE_HISTORY" : np.array([5, 10]),
    "NUM_LAYERS" : np.array([1, 5, 10]),
    "NUM_LAYERS_PRED" : np.array([1, 5, 10]),
    "HIDDEN_PRED" : np.array([50, 100,]),
    "LR" : np.array([1e-2, 1e-3]),
    "P" : np.array([0.1, 0.3]),
    "BALANCE_CLASSES" : np.array([False, True])}
    cv.set_grid(parameters)
    fold = int(sys.argv[1])
    print(f'fold {fold}')
    cv.perform_cv(fold=fold, n_epochs=1)

