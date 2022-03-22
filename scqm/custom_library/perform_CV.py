from scqm.custom_library.utils import DataPartition
from scqm.custom_library.preprocessing import load_dfs, preprocessing, extract_adanet_features
from scqm.custom_library.data_objects import Dataset
import itertools
import numpy as np
from adaptivenet_utils import train_model_on_partition
from scqm.custom_library.modules import Model
import sys
import csv
import random
import time
import torch

# seed = 0
# random.seed(seed)
# torch.manual_seed(seed)
# np.random.seed(seed)

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'fold {fold}')

    # instantiate model
    num_visit_features = dataset.visits_df_scaled_tensor_train.shape[1]
    num_medications_features = dataset.medications_df_scaled_tensor_train.shape[1]
    num_general_features = dataset.patients_df_scaled_tensor_train.shape[1]
    batch_first = True
    SIZE_EMBEDDING = np.array([30, 50])
    NUM_LAYERS_ENC = np.array([2, 5, 10])
    HIDDEN_ENC = np.array([50, 100])
    SIZE_HISTORY = np.array([30, 50])
    NUM_LAYERS = np.array([2,  5, 10])
    NUM_LAYERS_PRED = np.array([5, 10])
    HIDDEN_PRED = np.array([30, 50, 100])
    LR = np.array([1e-3])
    P = np.array([0.1, 0.2, 0.3])
    BALANCE_CLASSES = np.array([False, True])
    timestr = time.strftime("%Y%m%d-%H%M")
    task = 'classification'
    with open('/cluster/home/ctrottet/code/scqm_cv_results/' + timestr + '_fold_' + str(fold) + '.csv', 'w') as f:
        header = ['size_embedding', 'num_layers_enc','hidden_enc', 'size_history', 'num_layers', 'num_layers_pred', 'hidden_pred', 'p', 'lr', 'bal_classes','epochs', 'loss', 'loss_valid',
        'accuracy', 'accuracy_valid']
        writer = csv.writer(f)
        writer.writerow(header)
        combinations = list(itertools.product(SIZE_EMBEDDING, NUM_LAYERS_ENC, HIDDEN_ENC,
                            SIZE_HISTORY, NUM_LAYERS, NUM_LAYERS_PRED, HIDDEN_PRED, P, LR, BALANCE_CLASSES))
        combinations_sample = random.sample(combinations, 30)
        for ind, (size_embedding, num_layers_enc, hidden_enc, size_history, num_layers, num_layers_pred, hidden_pred, p, lr, bal) in enumerate(combinations_sample):
            print(f'{ind} combination out of {len(combinations_sample)}')
            print(
                f'size_embedding {size_embedding}, num_layers_enc {num_layers_enc},hidden_enc {hidden_enc}, size_history {size_history}, num_layers {num_layers}, num_layers_pred {num_layers_pred}, hidden_pred {hidden_pred}, dropout {p}, lr {lr}')
            model_specifics = {'size_embedding': size_embedding, 'num_layers_enc': num_layers_enc, 'hidden_enc': hidden_enc,  'size_history': size_history, 'num_layers': num_layers, 'num_layers_pred': num_layers_pred, 'hidden_pred': hidden_pred, 
                            'num_visit_features': num_visit_features, 'num_medications_features': num_medications_features, 'num_general_features': num_general_features, 'dropout' : p, 'batch_first': batch_first, 'device': device, 'task' : task}
            model = Model(model_specifics, device)
            
            model.set_training_parameters(n_epochs = 400, batch_size = 32, lr = lr, min_num_visits = 2, balance_classes = bal)
            e, loss, loss_valid, accuracy, accuracy_valid = train_model_on_partition(model, dataset, partition, fold, debug_patient = False)
            writer.writerow(np.array([size_embedding, num_layers_enc, hidden_enc, size_history,
                            num_layers, num_layers_pred, hidden_pred, p, lr, bal, e, loss, loss_valid, accuracy, accuracy_valid]))
