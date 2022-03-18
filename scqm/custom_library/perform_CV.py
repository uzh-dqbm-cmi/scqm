from scqm.custom_library.utils import DataPartition
from scqm.custom_library.preprocessing import load_dfs, preprocessing, extract_adanet_features
from scqm.custom_library.data_objects import Dataset
import itertools
import numpy as np
from adaptivenet_utils import train_model_on_partition, instantiate_model
import sys
import csv
import random
import time


if __name__ == '__main__':

    df_dict = load_dfs()
    df_dict = preprocessing(df_dict)
    patients_df, medications_df, visits_df, targets_df, _ = extract_adanet_features(df_dict, das28=True)
    df_dict_anet = {'visits': visits_df, 'patients': patients_df, 'medications': medications_df, 'targets': targets_df}
    dataset = Dataset(df_dict_anet, df_dict_anet['patients']['patient_id'].unique())
    # keep only patients with more than two visits
    dataset.drop([id_ for id_, patient in dataset.patients.items() if len(patient.visit_ids) <= 2])
    print(f'Dropping patients with less than 3 visits, keeping {len(dataset)}')
    # prepare for training
    dataset.transform_to_numeric_adanet()
    partition = DataPartition(dataset, k=5)
    print('start')
    fold = int(sys.argv[1])
    device = 'cuda:0'
    print(f'fold {fold}')

    # instantiate model
    num_visit_features = dataset.visits_df_scaled_tensor_train.shape[1]
    num_medications_features = dataset.medications_df_scaled_tensor_train.shape[1]
    num_general_features = dataset.patients_df_scaled_tensor_train.shape[1]
    batch_first = True
    SIZE_EMBEDDING = np.array([10, 30, 50])
    NUM_LAYERS_ENC = np.array([1, 2, 5, 10])
    HIDDEN_ENC = np.array([10, 30, 50])
    SIZE_HISTORY = np.array([10, 30, 50])
    NUM_LAYERS = np.array([1, 2,  5, 10])
    NUM_LAYERS_PRED = np.array([1, 5, 10, 15, 30])
    HIDDEN_PRED = np.array([10, 30, 50])
    LR = np.array([1e-4, 1e-3])
    timestr = time.strftime("%Y%m%d-%H%M")
    with open('/cluster/home/ctrottet/code/scqm_cv_results/' + timestr + '_fold_' + str(fold) + '.csv', 'w') as f:
        header = ['size_embedding', 'num_layers_enc','hidden_enc', 'size_history', 'num_layers', 'num_layers_pred', 'hidden_pred', 'lr', 'epochs', 'loss', 'loss_valid']
        writer = csv.writer(f)
        writer.writerow(header)
        combinations = list(itertools.product(SIZE_EMBEDDING, NUM_LAYERS_ENC, HIDDEN_ENC,
                            SIZE_HISTORY, NUM_LAYERS, NUM_LAYERS_PRED, HIDDEN_PRED, LR))
        combinations_sample = random.sample(combinations, 40)
        for ind, (size_embedding, num_layers_enc, hidden_enc, size_history, num_layers, num_layers_pred, hidden_pred, lr) in enumerate(combinations_sample):
            print(f'{ind} combination out of {len(combinations_sample)}')
            print(
                f'size_embedding {size_embedding}, num_layers_enc {num_layers_enc},hidden_enc {hidden_enc}, size_history {size_history}, num_layers {num_layers}, num_layers_pred {num_layers_pred}, hidden_pred {hidden_pred}, lr {lr}')
            model_specifics = {'size_embedding': size_embedding, 'num_layers_enc': num_layers_enc, 'hidden_enc': hidden_enc,  'size_history': size_history, 'num_layers': num_layers, 'num_layers_pred': num_layers_pred, 'hidden_pred': hidden_pred, 
                            'num_visit_features': num_visit_features, 'num_medications_features': num_medications_features, 'num_general_features': num_general_features, 'batch_first': batch_first, 'device': device}
            model, model_parameters = instantiate_model(
                model_specifics)
            parameters = {'data': dataset,
                        'partition': partition,
                        'fold': fold,
                        'device': device,
                        'model': model,
                        'model_parameters': model_parameters,
                        'batch_size': 32,
                        'epochs': 400,
                        'lr': lr,
                        'min_num_visits': 2,
                        'debug_patient': False}
            e, loss, loss_valid = train_model_on_partition(parameters)
            writer.writerow(np.array([size_embedding, num_layers_enc, hidden_enc, size_history,
                            num_layers, num_layers_pred, hidden_pred, lr, e, loss, loss_valid]))
