# patients with more than two visits and a minimum prediction horizon of 3 months up to a maximum of one year
# different history lengths of 6 months up to 5 years
# For the classification task DAS28-BSR>2.6 at next visit (mean interval 8.1 +-2.9 months from initial visit)
# TODO implement a filter that only selects the consultations from the last n years (5 in paper)

from scqm.custom_library.modules import *
from scqm.custom_library.preprocessing import *
from scqm.custom_library.utils import DataPartition
from scqm.custom_library.preprocessing import extract_adanet_features, preprocessing
from scqm.custom_library.data_objects import *
from scqm.custom_library.modules import VisitEncoder, MedicationEncoder, LSTMModule, PredModule
import matplotlib.pyplot as plt
from scqm.custom_library.utils import create_results_df

# # global parameters : network architecture
# # encoders
# hidden_1_enc = 20
# hidden_2_enc = 20
# size_embedding = 20
# # lstm
# size_history = 20
# # prediction module
# hidden_1_pred = 10
# hidden_2_pred = 10


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

# define training function


def get_batch_and_masks(all_indices, epoch, indices, dataset, batch_size, df_v, df_m, df_p, df_t, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, size_embedding, debug_patient=None):
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
    size = min(len(indices), batch_size)
    # batch and corresponding tensor indices
    batch = np.random.choice(indices, size=size, replace=False)

    # print(
    #     f'\ndebug id {debug_patient} \nnumber visits {len(dataset.patients[debug_patient].visits)} \n timeline {dataset.patients[debug_patient].timeline_visual}')

    # get corresponding indices in tensors

    tensor_indices_visits, tensor_indices_pat, tensor_indices_meds, tensor_indices_targ = [], [], [], []

    for elem in batch:
        tensor_indices_visits.extend(df_v[df_v.patient_id == elem]['tensor_indices_train'].values)
        tensor_indices_meds.extend(df_m[df_m.patient_id == elem]['tensor_indices_train'].values)
        tensor_indices_pat.extend(df_p[df_p.patient_id == elem]['tensor_indices_train'].values)
        tensor_indices_targ.extend(df_t[df_t.patient_id == elem]['tensor_indices_train'].values)
    tensor_indices_visits = np.array(tensor_indices_visits)
    tensor_indices_meds = np.array(tensor_indices_meds)
    tensor_indices_pat = np.array(tensor_indices_pat)
    tensor_indices_targ = np.array(tensor_indices_targ)
    # for debugging
    if debug_patient and debug_patient in batch:
        debug_index = list(batch).index(debug_patient)
        print(f'index in batch {debug_index}')
    else:
        debug_index = None
    if (tensor_indices_visits != tensor_indices_targ).any():
        raise ValueError('index mismatch between visits and targets')
    # corresponding tensor slices
    t_v = tensor_v[tensor_indices_visits]
    t_m = tensor_m[tensor_indices_meds]
    t_p = tensor_p[tensor_indices_pat]
    t_t = tensor_t[tensor_indices_targ]

    max_num_visits, seq_lengths, masks, visit_mask, total_num_visits_and_meds = get_masks(
        dataset, batch, min_num_visits, size_embedding, debug_patient)
    # remove batch indices from available indices (since one epoch is one pass through whole data set)
    indices = np.array([x for x in indices if x not in batch])
    # a whole pass through the data has been completed
    if len(indices) == 0:
        indices = all_indices
        epoch += 1

    return t_v, t_m, t_p, t_t, seq_lengths, masks, indices, epoch, max_num_visits, visit_mask, total_num_visits_and_meds, debug_index


def instantiate_model(model_specifics):
    """Instantiate the different modules of the model with the given parameters

    Args:
        num_visit_features (_type_): _description_
        num_medications_features (_type_): _description_
        num_general_features (_type_): _description_
        size_embedding (int, optional): _description_. Defaults to 10.
        size_history (int, optional): _description_. Defaults to 12.

    Returns:
        _type_: _description_
    """

    VEncoder = VisitEncoder(model_specifics['num_visit_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                            model_specifics['hidden_enc'])
    MEncoder = MedicationEncoder(model_specifics['num_medications_features'], model_specifics['size_embedding'],
                                 model_specifics['num_layers_enc'], model_specifics['hidden_enc'])
    LModule = LSTMModule(model_specifics['size_embedding'], model_specifics['device'],
                         model_specifics['batch_first'], model_specifics['size_history'], model_specifics['num_layers'])
    # + 1 for time to prediction
    PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
                         1, model_specifics['num_layers_pred'], model_specifics['hidden_pred'])
    model = {'visit_encoder': VEncoder.to(model_specifics['device']), 'medications_encoder': MEncoder.to(model_specifics['device']),
             'lstm': LModule.to(model_specifics['device']), 'pred_module': PModule.to(model_specifics['device'])}
    model_parameters = list(VEncoder.parameters()) + list(MEncoder.parameters()) + \
        list(LModule.parameters()) + list(PModule.parameters())
    return model, model_parameters


def train_model(dataset, device, model_specifics, batch_size=32, n_epochs=1, min_num_visits=2, debug_patient=False):
    #dfs and tensors
    df_v = dataset.visits_df_proc.copy()
    df_p = dataset.patients_df_proc.copy()
    df_m = dataset.medications_df_proc.copy()
    df_t = dataset.targets_df_proc.copy()
    tensor_v = dataset.visits_df_scaled_tensor_train.detach().clone().to(device)
    tensor_p = dataset.patients_df_scaled_tensor_train.detach().clone().to(device)
    tensor_m = dataset.medications_df_scaled_tensor_train.detach().clone().to(device)
    tensor_t = dataset.targets_df_scaled_tensor_train.detach().clone().to(device)
    # validation
    tensor_v_val = dataset.visits_df_scaled_tensor_valid.detach().clone().to(device)
    tensor_p_val = dataset.patients_df_scaled_tensor_valid.detach().clone().to(device)
    tensor_m_val = dataset.medications_df_scaled_tensor_valid.detach().clone().to(device)
    tensor_t_val = dataset.targets_df_scaled_tensor_valid.detach().clone().to(device)
    # instantiate modules
    num_visit_features = tensor_v.shape[1]
    num_medications_features = tensor_m.shape[1]
    num_general_features = tensor_p.shape[1]
    batch_first = True
    print(
        f'Starting training process with {num_visit_features} visit features, {num_medications_features} medication features, {num_general_features} general features and one time feature')
    
    model, model_parameters = instantiate_model(
        model_specifics)
    # model components
    VEncoder = model['visit_encoder']
    MEncoder = model['medications_encoder']
    LModule = model['lstm']
    PModule = model['pred_module']
    size_embedding = VEncoder.size_embedding
    # initial available indices and number of epochs
    all_indices = dataset.train_ids
    e = 0
    min_num_visits = 2
    optimizer = torch.optim.Adam(model_parameters, lr=1e-3)

    # debug patient
    if debug_patient:
        debug_patient = np.random.choice(all_indices, size=1)[0]
        print(
            f'Debug patient {debug_patient} \nall targets \n{df_t[df_t.patient_id == debug_patient]["das283bsr_score"]}')
    # validation data
    max_num_visits_val, seq_lengths_val, masks_val, visit_mask_val, total_num_val = get_masks(
        dataset, dataset.valid_ids, min_num_visits, size_embedding)
    # compute valid loss all valid_rate steps
    valid_rate = 10
    loss_per_epoch = torch.empty(size=(n_epochs, 1))
    loss_per_epoch_valid = torch.empty(size=(n_epochs // valid_rate, 1))
    indices = all_indices
    while (e < n_epochs):

        # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
        # patients with a given number of visits
        t_v, t_m, t_p, t_t, seq_lengths, masks, indices, e, max_num_visits, visit_mask, \
            total_num, debug_index = get_batch_and_masks(
                all_indices, e, indices, dataset, batch_size, df_v, df_m, df_p, df_t, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, size_embedding, debug_patient)
        # uncomment to check that dfs are the same
        # print(f'visits df first row debug patient {df_v[df_v.patient_id == debug_dict["id"]]}',
        # f'visits tensor first row {t_v[debug_dict["indices_in_tensors"][0]]}')
        # print(f'all targets for debug patient {t_t[debug_dict["indices_in_tensors"][3]]}')

        loss = apply_model_and_get_loss(device, VEncoder, MEncoder, LModule, PModule, t_v, t_m, t_p, t_t,
                                        min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug_index)

        # take optimizer step once loss wrt all visits has been computed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss and evaluate on validation data
        if len(indices) == len(dataset.train_ids):
            with torch.no_grad():
                loss_per_epoch[e - 1] = loss
                if e % valid_rate == 0:
                    loss_valid = apply_model_and_get_loss(device, VEncoder, MEncoder, LModule, PModule, tensor_v_val, tensor_m_val, tensor_p_val, tensor_t_val,
                                                          min_num_visits, max_num_visits_val, visit_mask_val, masks_val, seq_lengths_val, total_num_val)
                    loss_per_epoch_valid[e // valid_rate - 1] = loss_valid
                    print(f'epoch : {e} loss {loss} loss_valid {loss_valid}')

    #print(f'loss all epochs{loss_per_epoch} \n valid {loss_per_epoch_valid}')

    return loss_per_epoch, loss_per_epoch_valid, VEncoder, MEncoder, LModule, PModule


def train_model_on_partition(parameters):

    dataset = parameters['data']
    partition = parameters['partition']
    fold = parameters['fold']
    device = parameters['device']
    batch_size = parameters['batch_size']
    epochs = parameters['epochs']
    lr = parameters['lr']
    min_num_visits = parameters['min_num_visits']
    debug_patient = parameters['debug_patient']
    model = parameters['model']
    model_parameters = parameters['model_parameters']

    print(f'device {device}')
    #dfs and tensors
    df_v = dataset.visits_df_proc
    df_p = dataset.patients_df_proc
    df_m = dataset.medications_df_proc
    df_t = dataset.targets_df_proc
    tensor_v = dataset.visits_df_scaled_tensor_train.to(device)
    tensor_p = dataset.patients_df_scaled_tensor_train.to(device)
    tensor_m = dataset.medications_df_scaled_tensor_train.to(device)
    tensor_t = dataset.targets_df_scaled_tensor_train.to(device)

    # model components
    VEncoder = model['visit_encoder']
    MEncoder = model['medications_encoder']
    LModule = model['lstm']
    PModule = model['pred_module']
    size_embedding = VEncoder.size_embedding

    # validation

    tensor_indices_visits_val, tensor_indices_pat_val, tensor_indices_meds_val, tensor_indices_targ_val = [], [], [], []

    for elem in partition.partitions_test[fold]:
        tensor_indices_visits_val.extend(df_v[df_v.patient_id == elem]['tensor_indices_train'].values)
        tensor_indices_meds_val.extend(df_m[df_m.patient_id == elem]['tensor_indices_train'].values)
        tensor_indices_pat_val.extend(df_p[df_p.patient_id == elem]['tensor_indices_train'].values)
        tensor_indices_targ_val.extend(df_t[df_t.patient_id == elem]['tensor_indices_train'].values)
    tensor_indices_visits_val = np.array(tensor_indices_visits_val)
    tensor_indices_meds_val = np.array(tensor_indices_meds_val)
    tensor_indices_pat_val = np.array(tensor_indices_pat_val)
    tensor_indices_targ_val = np.array(tensor_indices_targ_val)
    tensor_v_val = tensor_v[tensor_indices_visits_val]
    tensor_p_val = tensor_p[tensor_indices_pat_val]
    tensor_m_val = tensor_m[tensor_indices_meds_val]
    tensor_t_val = tensor_t[tensor_indices_targ_val]
    max_num_visits_val, seq_lengths_val, masks_val, visit_mask_val, total_num_val = get_masks(
        dataset, partition.partitions_test[fold], min_num_visits, size_embedding)

    # initial available indices and number of epochs
    all_indices = partition.partitions_train[fold]
    e = 0
    min_num_visits = 2
    optimizer = torch.optim.Adam(model_parameters, lr=lr)

    # debug patient
    if debug_patient:
        debug_patient = np.random.choice(all_indices, size=1)[0]
        print(
            f'Debug patient {debug_patient} \nall targets \n{df_t[df_t.patient_id == debug_patient]["das283bsr_score"]}')

    loss_per_epoch = torch.empty(size=(epochs, 1))
    loss_per_epoch_valid = torch.empty(size=(epochs, 1))
    indices = all_indices
    early_stopping = False
    while (e < epochs) and early_stopping == False:

        # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
        # patients with a given number of visits
        t_v, t_m, t_p, t_t, seq_lengths, masks, indices, e, max_num_visits, visit_mask, \
            total_num, debug_index = get_batch_and_masks(all_indices,
                                                         e, indices, dataset, batch_size, df_v, df_m, df_p, df_t, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, size_embedding, debug_patient)
        # uncomment to check that dfs are the same
        # print(f'visits df first row debug patient {df_v[df_v.patient_id == debug_dict["id"]]}',
        # f'visits tensor first row {t_v[debug_dict["indices_in_tensors"][0]]}')
        # print(f'all targets for debug patient {t_t[debug_dict["indices_in_tensors"][3]]}')

        loss = apply_model_and_get_loss(device, VEncoder, MEncoder, LModule, PModule, t_v, t_m, t_p, t_t,
                                        min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug_index)

        # take optimizer step once loss wrt all visits has been computed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss and evaluate on validation data
        if len(indices) == len(all_indices):
            with torch.no_grad():
                loss_per_epoch[e - 1] = loss

                # # debug patient

                # debug_patient_valid = np.random.choice(partition.partitions_test[fold], size=1)[0]
                # debug_index_valid = list(partition.partitions_test[fold]).index(debug_patient_valid)
                # print(
                #     f'Debug patient {debug_patient_valid} \nall targets \n{df_t[df_t.patient_id == debug_patient_valid]["das283bsr_score"]}')
                loss_valid = apply_model_and_get_loss(device, VEncoder, MEncoder, LModule, PModule, tensor_v_val, tensor_m_val, tensor_p_val, tensor_t_val,
                                                      min_num_visits, max_num_visits_val, visit_mask_val, masks_val, seq_lengths_val, total_num_val)
                loss_per_epoch_valid[e - 1] = loss_valid
                print(f'epoch : {e} loss {loss} loss_valid {loss_valid}')
                if e > 20 and (loss_per_epoch_valid[e - 1 - 10:e - 1] / loss_per_epoch_valid[e - 1 - 20:e - 1 - 10]).mean() > 1:
                    early_stopping = True
                    print('valid loss increasing')

    return e, loss.item(), loss_valid.item()


def get_masks(dataset, subset, min_num_visits, size_embedding, debug_patient=None):
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
    max_num_visits = max([len(dataset.patients[index].visits) for index in subset])
    seq_lengths = torch.zeros(size=(max_num_visits - min_num_visits + 1,
                                    len(subset), 2), dtype=torch.long)
    # to store for each patient for each visit the visit/medication mask up to that visit. This mask allows
    # us to then easily combine the visit and medication events in the right order. True is for visit events and False for medications.
    # E.g. if a patient has the timeline [m1, m2, v1, m3, m4, v2, m5, v3] the corresponding masks up to each of the 3 visits would be
    # [[False, False], [False, False, True, False, False], [False, False, True, False, False, True, False]] and the sequence lengths
    # for visits/medication count up to each visit [[0, 2], [1, 4], [2, 5]]
    masks = [[] for i in range(len(subset))]
    for i, patient in enumerate(subset):
        for visit in range(0, len(dataset.patients[patient].visits) - min_num_visits + 1):
            # get timeline up to visit (not included)
            seq_lengths[visit, i, 0], seq_lengths[visit, i,
                                                  1], _, cropped_timeline_mask, visual = dataset.patients[patient].get_cropped_timeline(visit + min_num_visits)
            masks[i].append(torch.broadcast_to(torch.tensor([[tuple_[0]] for tuple_ in cropped_timeline_mask]),
                                               (len(cropped_timeline_mask), size_embedding)))
            if debug_patient and patient == debug_patient:
                print(f'visit {visit} cropped timeline mask {visual} ')

    # tensor of shape batch_size x max_num_visits with True in position (p, v) if patient p has at least v visits
    # and False else. we use this mask later to select the patients up to each visit.
    visit_mask = torch.tensor([[True if index <= len(dataset.patients[patient].visits)
                                else False for index in range(min_num_visits, max_num_visits + 1)] for patient in subset])

    # stores for each patient in batch the total number of visits and medications
    # it is used later to index correctly the visits and medications dataframes
    total_num_visits_and_meds = torch.tensor(
        [[len(dataset.patients[patient].visits), dataset.patients[patient].num_med_events] for patient in subset])
    return max_num_visits, seq_lengths, masks, visit_mask, total_num_visits_and_meds


def apply_model_and_get_loss(device, VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug=None):
    batch_first = True
    loss = 0
    criterion = torch.nn.MSELoss(reduction='sum')
    o_v, o_m = VEncoder(tensor_v), MEncoder(tensor_m)
    # for scaling of loss
    num_targets = 0
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
                    #print(f'visit info {tensor_v[index_visits:index_visits + seq[0]]} \n medication info {tensor_m[index_medications:index_medications + seq[1]]}')
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
        output, (hn, cn) = LModule(pack_padded_sequence)
        history = hn[-1]
        # concat computed patient history with general information
        pred_input = torch.cat((tensor_p[visit_mask[:, v]], history, time_to_targets), dim=1)
        # apply prediction module
        out = PModule(pred_input)
        # compute loss
        loss += criterion(out, targets)
        num_targets += len(targets)
        if debug_index_target != None:
            # print(out)
            print(f'prediction {out[debug_index_target].item()}')
    return loss / num_targets


def test_model(device, subset, dataset, VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug=None):

    batch_first = True
    loss = 0
    criterion = torch.nn.MSELoss()
    o_v, o_m = VEncoder(tensor_v), MEncoder(tensor_m)
    # 100 dummy value to indicate the visits that are not predicted for a given patient (i.e. all the visits > #num visits for this patient)
    results = torch.full(size=(len(subset), max_num_visits - min_num_visits + 1), fill_value=100.0, device=device)
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
        loss += criterion(out, targets)
        results[visit_mask[:, v], v] = out.flatten()
        if debug_index_target != None:
            # print(out)
            print(f'prediction {out[debug_index_target].item()}')
    results_df = create_results_df(device, subset, dataset, results)
    return results, results_df


def train_and_test_pipeline(dataset, batch_size, n_epochs, min_num_visits=2, debug = False):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')
    # train
    num_visit_features = dataset.visits_df_scaled_tensor_train.shape[1]
    num_medications_features = dataset.medications_df_scaled_tensor_train.shape[1]
    num_general_features = dataset.patients_df_scaled_tensor_train.shape[1]
    size_embedding = 10
    batch_first = True
    model_specifics = {'size_embedding': size_embedding, 'num_layers_enc': 2, 'hidden_enc': 20, 'size_history': 12, 'num_layers': 2, 'num_layers_pred': 2, 'hidden_pred': 20,
                       'num_visit_features': num_visit_features, 'num_medications_features': num_medications_features, 'num_general_features': num_general_features, 'batch_first': batch_first, 'device': device}
    loss_per_epoch, loss_per_epoch_valid, VEncoder, MEncoder, LModule, PModule = train_model(
        dataset, device, model_specifics, batch_size=batch_size, n_epochs=n_epochs, min_num_visits=min_num_visits, debug_patient=debug)

    # test
    with torch.no_grad():

        debug_patient, debug_index = None, None
        tensor_v_test = dataset.visits_df_scaled_tensor_test.detach().clone().to(device)
        tensor_p_test = dataset.patients_df_scaled_tensor_test.detach().clone().to(device)
        tensor_m_test = dataset.medications_df_scaled_tensor_test.detach().clone().to(device)
        tensor_t_test = dataset.targets_df_scaled_tensor_test.detach().clone().to(device)

        max_num_visits_test, seq_lengths_test, masks_test, visit_mask_test, total_num_test = get_masks(
            dataset, dataset.test_ids, min_num_visits, size_embedding, debug_patient=debug_patient)
        results, results_df = test_model(device, dataset.test_ids, dataset, VEncoder, MEncoder, LModule, PModule, tensor_v_test, tensor_m_test, tensor_p_test, tensor_t_test,
                                         min_num_visits, max_num_visits_test, visit_mask_test, masks_test, seq_lengths_test, total_num_test, debug=debug_index)

        # # test model on training data (just to see if predictions can get ~perfect)
        # #subset = np.random.choice(dataset.train_ids, 3)
        # subset = dataset.train_ids
        # # debug_patient = np.random.choice(subset, size=1)[0]
        # # debug_index = list(subset).index(debug_patient)
        # # df_t = dataset.targets_df_proc.copy()
        # # print(
        # #     f'Debug patient {debug_patient} \nall targets \n{df_t[df_t.patient_id == debug_patient]["das283bsr_score"]}')
        # tensor_v = dataset.visits_df_scaled_tensor_train.detach().clone().to(device)
        # tensor_p = dataset.patients_df_scaled_tensor_train.detach().clone().to(device)
        # tensor_m = dataset.medications_df_scaled_tensor_train.detach().clone().to(device)
        # tensor_t = dataset.targets_df_scaled_tensor_train.detach().clone().to(device)
        # max_num_visits, seq_lengths, masks, visit_mask, total_num = get_masks(
        #     dataset, subset, min_num_visits, size_embedding, debug_patient=debug_patient)
        # results_train, results_train_df = test_model(device, subset, dataset, VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t,
        #                                              min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug=debug_index)

    if len(loss_per_epoch_valid) > 0:
        plt.plot(range(0, len(loss_per_epoch), 1), loss_per_epoch)
        plt.plot(range(0, len(loss_per_epoch), int(len(loss_per_epoch) / len(loss_per_epoch_valid))), loss_per_epoch_valid)
        plt.ylim(0, 0.05)
    return loss_per_epoch, loss_per_epoch_valid, results_df