#patients with more than two visits and a minimum prediction horizon of 3 months up to a maximum of one year
#different history lengths of 6 months up to 5 years
# For the classification task DAS28-BSR>2.6 at next visit (mean interval 8.1 +-2.9 months from initial visit)
#TODO implement a filter that only selects the consultations from the last n years (5 in paper)

from scqm.custom_library.modules import *
from scqm.custom_library.preprocessing import *
import os
from scqm.custom_library.preprocessing import extract_adanet_features, preprocessing
from scqm.custom_library.data_objects import *
from scqm.custom_library.modules import VisitEncoder, MedicationEncoder, LSTMModule, PredModule
import matplotlib.pyplot as plt

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
        

def get_batch_and_masks(epoch, indices, dataset, batch_size, df_v, df_m, df_p, df_t, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, debug_patient=None):
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
    else : 
        debug_index = None
    if (tensor_indices_visits != tensor_indices_targ).any():
        raise ValueError('index mismatch between visits and targets')
    # corresponding tensor slices
    t_v = tensor_v[tensor_indices_visits]
    t_m = tensor_m[tensor_indices_meds]
    t_p = tensor_p[tensor_indices_pat]
    t_t = tensor_t[tensor_indices_targ]

    max_num_visits, seq_lengths, masks, visit_mask, total_num_visits_and_meds = get_masks(dataset, batch, min_num_visits, debug_patient)
    # remove batch indices from available indices (since one epoch is one pass through whole data set)
    indices = np.array([x for x in indices if x not in batch])
    # a whole pass through the data has been completed
    if len(indices) == 0:
        indices = dataset.train_ids
        epoch += 1

    return t_v, t_m, t_p, t_t, seq_lengths, masks, indices, epoch, max_num_visits, visit_mask, total_num_visits_and_meds, debug_index

def instantiate_model(num_visit_features, num_medications_features, num_general_features, batch_first, 
size_embedding=10, size_history=12):
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

    VEncoder = VisitEncoder(num_visit_features, size_embedding)
    MEncoder = MedicationEncoder(num_medications_features, size_embedding)
    LModule = LSTMModule(size_embedding, batch_first, size_history)
    # + 1 for time to prediction
    PModule = PredModule(size_history + num_general_features + 1)
    model = {'visit_encoder': VEncoder, 'medications_encoder': MEncoder,
             'lstm': LModule, 'pred_module': PModule}
    model_parameters = list(VEncoder.parameters()) + list(MEncoder.parameters()) + list(LModule.parameters()) + list(PModule.parameters())
    return model, model_parameters

def train_model(dataset, device = torch.device('cpu'), batch_size=32, n_epochs =1, min_num_visits = 2, debug_patient = False):
    #dfs and tensors
    df_v = dataset.visits_df_proc.copy()
    df_p = dataset.patients_df_proc.copy()
    df_m = dataset.medications_df_proc.copy()
    df_t = dataset.targets_df_proc.copy()
    tensor_v = dataset.visits_df_scaled_tensor_train.detach().clone().to(device)
    tensor_p = dataset.patients_df_scaled_tensor_train.detach().clone().to(device)
    tensor_m = dataset.medications_df_scaled_tensor_train.detach().clone().to(device)
    tensor_t = dataset.targets_df_scaled_tensor_train.detach().clone().to(device)
    #validation
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
    model, model_parameters = instantiate_model(num_visit_features, num_medications_features, num_general_features, batch_first)
    #model components
    VEncoder = model['visit_encoder']
    MEncoder = model['medications_encoder']
    LModule = model['lstm']
    PModule = model['pred_module']
    VEncoder.to(device)
    MEncoder.to(device)
    LModule.to(device)
    PModule.to(device)
    # initial available indices and number of epochs
    indices = dataset.train_ids
    e = 0
    min_num_visits = 2
    optimizer = torch.optim.Adam(model_parameters, lr = 1e-3)

    # debug patient
    if debug_patient :
        debug_patient = np.random.choice(indices, size=1)[0]
        print(f'Debug patient {debug_patient} \nall targets \n{df_t[df_t.patient_id == debug_patient]["das283bsr_score"]}')
    #validation data
    max_num_visits_val, seq_lengths_val, masks_val, visit_mask_val, total_num_val = get_masks(
        dataset, dataset.valid_ids, min_num_visits)
    # compute valid loss all valid_rate steps
    valid_rate = 10
    loss_per_epoch = torch.empty(size = (n_epochs, 1))
    loss_per_epoch_valid = torch.empty(size=(n_epochs//valid_rate, 1))
    while (e < n_epochs):


        # get batch, corresponding tensor slices and masks to combine the visits/medication events and to select the
        # patients with a given number of visits
        t_v, t_m, t_p, t_t, seq_lengths, masks, indices, e, max_num_visits, visit_mask, \
            total_num, debug_index = get_batch_and_masks(e, indices, dataset, batch_size, df_v, df_m, df_p, df_t, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, debug_patient)
        # uncomment to check that dfs are the same
        # print(f'visits df first row debug patient {df_v[df_v.patient_id == debug_dict["id"]]}',
        # f'visits tensor first row {t_v[debug_dict["indices_in_tensors"][0]]}')
        # print(f'all targets for debug patient {t_t[debug_dict["indices_in_tensors"][3]]}')
        
        loss = apply_model_and_get_loss(VEncoder, MEncoder, LModule, PModule, t_v, t_m, t_p, t_t,
                                        min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug_index)

        # take optimizer step once loss wrt all visits has been computed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss and evaluate on validation data
        if len(indices) == len(dataset.train_ids):
            with torch.no_grad():
                loss_per_epoch[e-1] = loss
                if e%valid_rate == 0:
                    loss_valid = apply_model_and_get_loss(VEncoder, MEncoder, LModule, PModule, tensor_v_val, tensor_m_val, tensor_p_val, tensor_t_val,
                                        min_num_visits, max_num_visits_val, visit_mask_val, masks_val, seq_lengths_val, total_num_val)
                    loss_per_epoch_valid[e//valid_rate-1] = loss_valid
                    print(f'epoch : {e} loss {loss} loss_valid {loss_valid}')


    #print(f'loss all epochs{loss_per_epoch} \n valid {loss_per_epoch_valid}')


    return loss_per_epoch, loss_per_epoch_valid, VEncoder, MEncoder, LModule, PModule


def get_masks(dataset, subset, min_num_visits, debug_patient=None):
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
                                                (len(cropped_timeline_mask), 10)))
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

def apply_model_and_get_loss(VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug = None):
    batch_first = True
    loss = 0
    criterion = torch.nn.MSELoss()
    o_v, o_m = VEncoder(tensor_v), MEncoder(tensor_m)
    for v in range(0, max_num_visits - min_num_visits + 1):
        # stores for all the patients in the batch the tensor of ordered events (of varying size)
        sequence = []
        # to keep track of the right index in the visits/medication tensors
        index_visits = 0
        index_medications = 0
        # targets
        targets = torch.empty(size=(torch.sum(visit_mask[:, v] == True).item(), 1))
        # delta t
        time_to_targets = torch.empty(size=(torch.sum(visit_mask[:, v] == True).item(), 1))
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
        if debug_index_target != None:
            #print(out)
            print(f'prediction {out[debug_index_target].item()}')
    return loss


def test_model(dataset, VEncoder, MEncoder, LModule, PModule, tensor_v, tensor_m, tensor_p, tensor_t, min_num_visits, max_num_visits, visit_mask, masks, seq_lengths, total_num, debug=None):
    batch_first = True
    loss = 0
    criterion = torch.nn.MSELoss()
    o_v, o_m = VEncoder(tensor_v), MEncoder(tensor_m)
    results = torch.full(size=(len(dataset.test_ids), max_num_visits - min_num_visits + 1), fill_value=100.0)
    for v in range(0, max_num_visits - min_num_visits + 1):
        # stores for all the patients in the batch the tensor of ordered events (of varying size)
        sequence = []
        # to keep track of the right index in the visits/medication tensors
        index_visits = 0
        index_medications = 0
        # targets
        targets = torch.empty(size=(torch.sum(visit_mask[:, v] == True).item(), 1))
        # delta t
        time_to_targets = torch.empty(size=(torch.sum(visit_mask[:, v] == True).item(), 1))
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
        results[visit_mask[:, v], v] = out.flatten()
        if debug_index_target != None:
            #print(out)
            print(f'prediction {out[debug_index_target].item()}')
    results_df = create_results_df(dataset, results)
    return results, results_df


def train_and_test_pipeline(dataset, batch_size, n_epochs, min_num_visits=2):
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')
    #train
    loss_per_epoch, loss_per_epoch_valid, VEncoder, MEncoder, LModule, PModule = train_model(
        dataset, batch_size=batch_size, n_epochs=n_epochs, min_num_visits=min_num_visits, debug_patient=False)
    #test
    with torch.no_grad():
        tensor_v_test = dataset.visits_df_scaled_tensor_test.detach().clone().to(device)
        tensor_p_test = dataset.patients_df_scaled_tensor_test.detach().clone().to(device)
        tensor_m_test = dataset.medications_df_scaled_tensor_test.detach().clone().to(device)
        tensor_t_test = dataset.targets_df_scaled_tensor_test.detach().clone().to(device)

        max_num_visits_test, seq_lengths_test, masks_test, visit_mask_test, total_num_test = get_masks(
            dataset, dataset.test_ids, min_num_visits)
        results, results_df = test_model(dataset, VEncoder, MEncoder, LModule, PModule, tensor_v_test, tensor_m_test, tensor_p_test, tensor_t_test,
                                         min_num_visits, max_num_visits_test, visit_mask_test, masks_test, seq_lengths_test, total_num_test)
    
    if len(loss_per_epoch_valid) > 0:
        plt.plot(range(0, len(loss_per_epoch), 1), loss_per_epoch)
        plt.plot(range(0, len(loss_per_epoch), int(len(loss_per_epoch) / len(loss_per_epoch_valid))), loss_per_epoch_valid)
    return loss_per_epoch, loss_per_epoch_valid, results, results_df

def create_results_df(dataset, results):
    tmp_1 = dataset.targets_df[dataset.targets_df.patient_id.isin(dataset.test_ids)].copy()
    tmp_2 = dataset.targets_df_proc[dataset.targets_df_proc.patient_id.isin(
        dataset.test_ids)][['patient_id', 'visit_date', 'das283bsr_score']]
    tmp_3 = pd.concat([tmp_1, tmp_2.rename(columns={'visit_date': 'visit_date_scaled', 'das283bsr_score':'das28_scaled'})], axis=1)
    results_df = tmp_3.loc[:, ~tmp_3.columns.duplicated()]
    for index, elem in enumerate(results_df['patient_id'].unique()):
        predictions = [np.nan] + [value for value in results[index, :] if value != 100.0]
        results_df.loc[results_df.patient_id == elem, 'scaled_predictions'] = predictions
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
    f1 = plt.figure()
    plt.scatter(results_df['days_to_prev_visit'], results_df['squarred_error'], marker='x', alpha=0.5)
    plt.xlabel('days')
    plt.ylabel('Squarred error')
    plt.xlim(0,1000)
    f2 = plt.figure()
    plt.scatter(results_df['history_length'], results_df['squarred_error'], marker='x', alpha=0.5)

    plt.xlabel('history length')
    plt.ylabel('Squarred error')

    return


