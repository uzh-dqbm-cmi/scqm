# patients with more than two visits and a minimum prediction horizon of 3 months up to a maximum of one year
# different history lengths of 6 months up to 5 years
# For the classification task DAS28-BSR>2.6 at next visit (mean interval 8.1 +-2.9 months from initial visit)
# TODO implement a filter that only selects the consultations from the last n years (5 in paper)

from scqm.custom_library.modules import *
from scqm.custom_library.preprocessing import *
from scqm.custom_library.utils import DataPartition, create_results_df
from scqm.custom_library.preprocessing import extract_adanet_features, preprocessing
from scqm.custom_library.data_objects import *
from scqm.custom_library.modules import VisitEncoder, MedicationEncoder, LSTMModule, PredModule
import matplotlib.pyplot as plt

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


def KFoldCVpipeline(df_dict):
    df_dict = preprocessing(df_dict)
    patients_df, medications_df, visits_df, targets_df, _ = extract_adanet_features(df_dict, das28=True)
    df_dict_anet = {'visits': visits_df, 'patients': patients_df, 'medications': medications_df, 'targets': targets_df}
    dataset = Dataset(df_dict_anet, df_dict_anet['patients']['patient_id'].unique())
    # keep only patients with more than two visits
    dataset.drop([id_ for id_, patient in dataset.patients.items() if len(patient.visit_ids) <= 2])
    print(f'Dropping patients with less than 3 visits, keeping {len(dataset)}')
    # prepare for training
    dataset.transform_to_numeric_adanet()
    # partition for CV
    partition = DataPartition(dataset, k=5)
    partition.split()

    return dataset, partition


def create_dataset(df_dict):
    df_dict = preprocessing(df_dict)
    patients_df, medications_df, visits_df, targets_df, _ = extract_adanet_features(df_dict, das28=True)
    df_dict_anet = {'visits': visits_df, 'patients': patients_df, 'medications': medications_df, 'targets': targets_df}
    dataset = Dataset(df_dict_anet, df_dict_anet['patients']['patient_id'].unique())
    # keep only patients with more than two visits
    dataset.drop([id_ for id_, patient in dataset.patients.items() if len(patient.visit_ids) <= 2])
    print(f'Dropping patients with less than 3 visits, keeping {len(dataset)}')
    # prepare for training
    dataset.transform_to_numeric_adanet()
    dataset.split_data()
    dataset.scale_and_tensor()

    return dataset


def test_model(task, device, subset, dataset, VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug=None):

    batch_first = True
    loss = 0
    #criterion = torch.nn.MSELoss()
    o_v, o_m = VEncoder(tensor_v), MEncoder(tensor_m)
    # 100 dummy value to indicate the visits that are not predicted for a given patient (i.e. all the visits > #num visits for this patient)
    if task == 'regression':
        results = torch.full(size=(len(subset), max_num_visits - min_num_visits + 1), fill_value=100.0, device=device)
    else:
        results = torch.full(size=(len(subset), max_num_visits - min_num_visits + 1),
                             fill_value=100.0, device=device, dtype=torch.int64)
    for v in range(0, max_num_visits - min_num_visits + 1):
        # stores for all the patients in the batch the tensor of ordered events (of varying size)
        sequence = []
        # to keep track of the right index in the visits/medication tensors
        index_visits = 0
        index_medications = 0
        # targets
        targets = torch.empty(size=(torch.sum(visit_mask[:, v] == True).item(), 1), device=device)
        # delta t
        time_to_targets = torch.empty(size=(torch.sum(visit_mask[:, v] == True).item(), 1), device=device)
        # for each patient combine the medication and visit events in the right order up to visit v
        index_target = 0
        debug_index_target = None

        for patient, seq in enumerate(seq_lengths[v]):
            # check if the patient has at least v visits
            if visit_mask[patient, v] == True:
                # create combined ordered list of visit/medication events up to v
                combined = torch.zeros_like(torch.cat([o_v[index_visits:index_visits + seq[0]],
                                                       o_m[index_medications:index_medications + seq[1]]]))
                combined[masks[patient][v]] = o_v[index_visits:index_visits + seq[0]].flatten()
                combined[~masks[patient][v]] = o_m[index_medications:index_medications + seq[1]].flatten()

                sequence.append(combined)
                targets[index_target] = tensor_t[index_visits + seq[0], 1]
                time_to_targets[index_target] = tensor_t[index_visits + seq[0], 0]
                if debug != None and patient == debug:
                    print(f'next target : {targets[index_target]}')
                    debug_index_target = index_target
                index_target += 1
            # update the indices to select from in the tensors
            index_visits += total_num[patient, 0]
            index_medications += total_num[patient, 1]
        # "preprocessing" to apply lstm
        padded_sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=batch_first)
        # compute the lengths of the sequences for each patient with available visit v
        lengths = seq_lengths[v].sum(dim=1)[visit_mask[:, v]]

        pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            padded_sequence, batch_first=batch_first, lengths=lengths, enforce_sorted=False)
        # apply lstm
        LModule.to(device)
        output, (hn, cn) = LModule(pack_padded_sequence)
        history = hn[-1]
        # concat computed patient history with general information
        pred_input = torch.cat((tensor_p[visit_mask[:, v]], history, time_to_targets), dim=1)
        # apply prediction module
        out = PModule(pred_input)
        # compute loss
        # loss += criterion(out, targets)
        if task == 'regression':
            results[visit_mask[:, v], v] = out.flatten()
        else:
            results[visit_mask[:, v], v] = torch.tensor([1 if elem >= 0.5 else 0 for elem in out])
        if debug_index_target != None:
            # print(out)
            print(f'prediction {out[debug_index_target].item()}')
    results_df = create_results_df(device, subset, dataset, results)
    return results, results_df


# def train_and_test_pipeline(dataset, batch_size, n_epochs, task='regression', min_num_visits=2, debug=False):
#     # device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f'device {device}')
#     # train
#     num_visit_features = dataset.visits_df_scaled_tensor_train.shape[1]
#     num_medications_features = dataset.medications_df_scaled_tensor_train.shape[1]
#     num_general_features = dataset.patients_df_scaled_tensor_train.shape[1]
#     size_embedding = 50
#     batch_first = True
#     model_specifics = {'task': task, 'size_embedding': size_embedding, 'num_layers_enc': 1, 'hidden_enc': 10, 'size_history': 10, 'num_layers': 2, 'num_layers_pred': 1, 'hidden_pred': 30,
#                        'num_visit_features': num_visit_features, 'num_medications_features': num_medications_features, 'num_general_features': num_general_features, 'batch_first': batch_first, 'device': device, 'dropout': 0.1,
#                        'balance_classes': True}
#     e, loss_per_epoch, loss_per_epoch_valid, accuracy_per_epoch, accuracy_per_epoch_valid, VEncoder, MEncoder, LModule, PModule = train_model(
#         dataset, device, model_specifics, batch_size=batch_size, n_epochs=n_epochs, min_num_visits=min_num_visits, debug_patient=debug)

#     # test
#     with torch.no_grad():

#         debug_patient, debug_index = None, None
#         tensor_v_test = dataset.visits_df_scaled_tensor_test.detach().clone().to(device)
#         tensor_p_test = dataset.patients_df_scaled_tensor_test.detach().clone().to(device)
#         tensor_m_test = dataset.medications_df_scaled_tensor_test.detach().clone().to(device)
#         tensor_t_test = dataset.targets_df_scaled_tensor_test.detach().clone().to(device)

#         max_num_visits_test, seq_lengths_test, masks_test, visit_mask_test, total_num_test = get_masks(
#             dataset, dataset.test_ids, min_num_visits, size_embedding, debug_patient=debug_patient)
#         VEncoder.eval()
#         MEncoder.eval()
#         LModule.eval()
#         PModule.eval()
#         results, results_df = test_model(task, device, dataset.test_ids, dataset, VEncoder, MEncoder, LModule, PModule, tensor_v_test, tensor_m_test, tensor_p_test, tensor_t_test,
#                                          min_num_visits, max_num_visits_test, visit_mask_test, masks_test, seq_lengths_test, total_num_test, debug=debug_index)

#         # # test model on training data (just to see if predictions can get ~perfect)
#         # #subset = np.random.choice(dataset.train_ids, 3)
#         # subset = dataset.train_ids
#         # # debug_patient = np.random.choice(subset, size=1)[0]
#         # # debug_index = list(subset).index(debug_patient)
#         # # df_t = dataset.targets_df_proc.copy()
#         # # print(
#         # #     f'Debug patient {debug_patient} \nall targets \n{df_t[df_t.patient_id == debug_patient]["das283bsr_score"]}')
#         # tensor_v = dataset.visits_df_scaled_tensor_train.detach().clone().to(device)
#         # tensor_p = dataset.patients_df_scaled_tensor_train.detach().clone().to(device)
#         # tensor_m = dataset.medications_df_scaled_tensor_train.detach().clone().to(device)
#         # tensor_t = dataset.targets_df_scaled_tensor_train.detach().clone().to(device)
#         # max_num_visits, seq_lengths, masks, visit_mask, total_num = get_masks(
#         #     dataset, subset, min_num_visits, size_embedding, debug_patient=debug_patient)
#         # results_train, results_train_df = test_model(device, subset, dataset, VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t,
#         #                                              min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug=debug_index)

#     if len(loss_per_epoch_valid) > 0:
#         plt.plot(range(0, len(loss_per_epoch[:e]), 1), loss_per_epoch[:e])
#         plt.plot(range(0, len(loss_per_epoch[:e]), 1), loss_per_epoch_valid[:e])
#         if task == 'regression':
#             plt.ylim(0, 0.05)
#         plt.figure()
#         plt.plot(range(0, len(accuracy_per_epoch[:e]), 1), accuracy_per_epoch[:e])
#         plt.plot(range(0, len(accuracy_per_epoch[:e]), 1), accuracy_per_epoch_valid[:e])

#     return loss_per_epoch, loss_per_epoch_valid, results_df

#TODO implement test and plot
