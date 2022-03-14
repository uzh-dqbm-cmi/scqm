from scqm.custom_library.preprocessing import *
from scqm.custom_library.data_objects import *
from scqm.custom_library.modules import MLP

import matplotlib.pyplot as plt

def create_dataset_baselines(df_dict):
    df_dict = preprocessing(df_dict)
    patients_df, medications_df, visits_df, targets_df, joint_df = extract_adanet_features(df_dict, das28=True, joint_df=True)
    df_dict_baselines = {'visits': visits_df, 'patients': patients_df, 'medications': medications_df, 'targets': targets_df, 'joint' : joint_df}
    dataset = Dataset(df_dict_baselines, df_dict_baselines['patients']['patient_id'].unique())
    # keep only patients with more than two visits
    dataset.drop([id_ for id_, patient in dataset.patients.items() if len(patient.visit_ids) <= 2])
    print(f'Dropping patients with less than 3 visits, keeping {len(dataset)}')
    # prepare for training
    dataset.transform_to_numeric_adanet()
    dataset.split_data()
    dataset.scale_and_tensor()

    return dataset


def train_model(dataset, batch_size=32, n_epochs=1, hist_size = 10, device=torch.device('cpu'), debug_patient = True):
    #dfs and tensors
    feature_df = dataset.joint_df_proc.copy()
    patient_df = dataset.patients_df_proc.copy()
    feature_tensor = dataset.joint_df_scaled_tensor_train.detach().clone().to(device)
    patient_tensor = dataset.patients_df_scaled_tensor_train.detach().clone().to(device)
    #validation
    feature_tensor_val = dataset.joint_df_scaled_tensor_valid.detach().clone().to(device)
    patient_tensor_val = dataset.patients_df_scaled_tensor_valid.detach().clone().to(device)
    input_tensor_valid, targets_valid = get_input_tensors(dataset.valid_ids, 'valid', feature_df, patient_df, feature_tensor_val, patient_tensor_val, hist_size, debug_patient = None)
    
    num_features = feature_tensor.shape[1] *  hist_size+ patient_tensor.shape[1] + 1

    #model components
    MLPmodel = MLP(num_features)

    # initial available indices and number of epochs
    indices = dataset.train_ids
    # debug patient
    if debug_patient:
        debug_patient = np.random.choice(indices, size=1)[0]
        print(
            f'Debug patient {debug_patient} \nall targets \n{feature_df[(feature_df.patient_id == debug_patient) & (feature_df.is_visit == 1)]["das283bsr_score"]}')
    e = 0

    optimizer = torch.optim.Adam(MLPmodel.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    # compute valid loss all valid_rate steps
    valid_rate = 10
    loss_per_epoch = torch.empty(size=(n_epochs, 1))
    loss_per_epoch_valid = torch.empty(size=(n_epochs // valid_rate, 1))
    while (e < n_epochs):

        input_tensor, targets, e, indices = get_batch(
                e, indices, dataset, batch_size, feature_df, patient_df, feature_tensor, patient_tensor, hist_size, debug_patient)


        out =  MLPmodel(input_tensor)
        loss = criterion(out, targets)
        # take optimizer step once loss wrt all visits has been computed
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # store loss and evaluate on validation data
        if len(indices) == len(dataset.train_ids):
            with torch.no_grad():
                loss_per_epoch[e - 1] = loss
                if e % valid_rate == 0:
                    out_valid = MLPmodel(input_tensor_valid)
                    loss_valid = criterion(out_valid, targets_valid)
                    loss_per_epoch_valid[e // valid_rate - 1] = loss_valid
                    print(f'epoch : {e} loss {loss} loss_valid {loss_valid}')

    #print(f'loss all epochs{loss_per_epoch} \n valid {loss_per_epoch_valid}')

    return loss_per_epoch, loss_per_epoch_valid


def get_batch(e, indices, dataset, batch_size, feature_df, patient_df, feature_tensor, patient_tensor, hist_size, debug_patient=None):

    # batch size
    size = min(len(indices), batch_size)

    batch = np.random.choice(indices, size=size, replace=False)
    input_tensor, input_targets = get_input_tensors(
        batch, 'train', feature_df, patient_df, feature_tensor, patient_tensor, hist_size, debug_patient)
    
    indices = np.array([x for x in indices if x not in batch])
    if len(indices) == 0:
        indices = dataset.train_ids
        e += 1


    return input_tensor, input_targets, e, indices 

def get_input_tensors(subset, subset_name, feature_df, patient_df, feature_tensor, patient_tensor,  hist_size, debug_patient):
    #TODO make this more robust
    visit_flag = [col for col in feature_df.columns if col not in ['patient_id', 'uid_num', 'med_id']].index('is_visit')
    target_flag = [col for col in feature_df.columns if col not in [
        'patient_id', 'uid_num', 'med_id']].index('das283bsr_score')
    visit_date_flag = [col for col in feature_df.columns if col not in [
        'patient_id', 'uid_num', 'med_id']].index('date')
    input_tensor = torch.Tensor()
    input_targets = torch.Tensor()

    for elem in subset:
        if subset_name == 'train':
            feature_indices = feature_df[feature_df.patient_id == elem]['tensor_indices_train'].values
            general_indices = patient_df[patient_df.patient_id == elem]['tensor_indices_train'].values
        elif subset_name == 'valid':
            feature_indices = feature_df[feature_df.patient_id == elem]['tensor_indices_valid'].values
            general_indices = patient_df[patient_df.patient_id == elem]['tensor_indices_valid'].values
        t = feature_tensor[feature_indices]
        p = patient_tensor[general_indices]
        # transform
        indices_of_visits = torch.tensor([elem for elem in range(len(t)) if t[elem, visit_flag] == 1])
        # not enough rows
        if indices_of_visits[-1] < hist_size + 1:
            if elem == debug_patient:
                print(
                    f'history too short for debug patient, length of history before last visit {indices_of_visits[-1]-1} minimum required length {hist_size} appending {hist_size + 1 - indices_of_visits[-1]} rows')
            t = torch.cat([torch.full(size=(hist_size + 1 - indices_of_visits[-1], t.shape[1]), fill_value=-1), t], axis=0)
            indices_of_visits += (hist_size + 1 - indices_of_visits[-1])
        visits_to_predict = torch.tensor([elem for elem in indices_of_visits if elem > hist_size])
        tensor_x = torch.empty(size=(len(visits_to_predict), t.shape[1] * hist_size + p.shape[1] + 1))
        tensor_y = torch.empty(size=(len(visits_to_predict), 1))
        for index, visit in enumerate(visits_to_predict):
            tensor_x[index] = torch.cat([t[visit - hist_size:visit].flatten(), p.flatten(), t[visit, visit_date_flag].reshape(1)])
            tensor_y[index] = t[visit, target_flag].reshape(1, 1)
        input_tensor = torch.cat([input_tensor, tensor_x])
        input_targets = torch.cat([input_targets, tensor_y])
        if elem == debug_patient:
            print(f'targets to predict : {tensor_y}')

    return input_tensor, input_targets


def train_and_test_pipeline_baseline(dataset, batch_size, n_epochs, hist_size):
    # device
    #TODO implement debug patient for baseline
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')
    #train
    loss_per_epoch, loss_per_epoch_valid = train_model(
        dataset, batch_size, n_epochs, hist_size, device)
    # #test
    # with torch.no_grad():
    #     tensor_v_test = dataset.visits_df_scaled_tensor_test.detach().clone().to(device)
    #     tensor_p_test = dataset.patients_df_scaled_tensor_test.detach().clone().to(device)
    #     tensor_m_test = dataset.medications_df_scaled_tensor_test.detach().clone().to(device)
    #     tensor_t_test = dataset.targets_df_scaled_tensor_test.detach().clone().to(device)

    #     max_num_visits_test, seq_lengths_test, masks_test, visit_mask_test, total_num_test = get_masks(
    #         dataset, dataset.test_ids, min_num_visits)
    #     results, results_df = test_model(dataset, VEncoder, MEncoder, LModule, PModule, tensor_v_test, tensor_m_test, tensor_p_test, tensor_t_test,
    #                                      min_num_visits, max_num_visits_test, visit_mask_test, masks_test, seq_lengths_test, total_num_test)

    if len(loss_per_epoch_valid) > 0:
        plt.plot(range(0, len(loss_per_epoch), 1), loss_per_epoch)
        plt.plot(range(0, len(loss_per_epoch), int(len(loss_per_epoch) / len(loss_per_epoch_valid))), loss_per_epoch_valid)
    return loss_per_epoch, loss_per_epoch_valid
