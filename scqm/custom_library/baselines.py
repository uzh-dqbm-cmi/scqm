from scqm.custom_library.utils import create_results_df
from scqm.custom_library.preprocessing import *
from scqm.custom_library.data_objects import *
from scqm.custom_library.modules import MLP

import matplotlib.pyplot as plt

#global parameters : network architecture
hidden_1_mlp = 20
hidden_2_mlp = 20

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
    input_tensor_valid, targets_valid,_ = get_input_tensors(device, dataset.valid_ids, 'valid', feature_df, patient_df, feature_tensor_val, patient_tensor_val, hist_size, debug_patient = None)
    
    num_features = feature_tensor.shape[1] *  hist_size+ patient_tensor.shape[1] + 1

    #model components
    MLPmodel = MLP(num_features, hidden_1_mlp, hidden_2_mlp)
    MLPmodel.to(device)

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

        input_tensor, targets, e, indices = get_batch(device, 
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

    return loss_per_epoch, loss_per_epoch_valid, MLPmodel


def get_batch(device, e, indices, dataset, batch_size, feature_df, patient_df, feature_tensor, patient_tensor, hist_size, debug_patient=None):

    # batch size
    size = min(len(indices), batch_size)

    batch = np.random.choice(indices, size=size, replace=False)
    input_tensor, input_targets,_ = get_input_tensors(device, 
        batch, 'train', feature_df, patient_df, feature_tensor, patient_tensor, hist_size, debug_patient)
    
    indices = np.array([x for x in indices if x not in batch])
    if len(indices) == 0:
        indices = dataset.train_ids
        e += 1


    return input_tensor, input_targets, e, indices 

def get_input_tensors(device, subset, subset_name, feature_df, patient_df, feature_tensor, patient_tensor,  hist_size, debug_patient, method = 'predict_all_visits'):
    #TODO make this more robust
    visit_flag = [col for col in feature_df.columns if col not in ['patient_id', 'uid_num', 'med_id']].index('is_visit')
    target_flag = [col for col in feature_df.columns if col not in [
        'patient_id', 'uid_num', 'med_id']].index('das283bsr_score')
    visit_date_flag = [col for col in feature_df.columns if col not in [
        'patient_id', 'uid_num', 'med_id']].index('date')
    input_tensor = torch.empty((0), device = device)
    input_targets = torch.empty((0), device = device)
    # keep track of number of targets for each element 
    num_targets = np.empty(len(subset))
    for index_in_subset, elem in enumerate(subset):
        if subset_name == 'train':
            feature_indices = feature_df[feature_df.patient_id == elem]['tensor_indices_train'].values
            general_indices = patient_df[patient_df.patient_id == elem]['tensor_indices_train'].values
        elif subset_name == 'valid':
            feature_indices = feature_df[feature_df.patient_id == elem]['tensor_indices_valid'].values
            general_indices = patient_df[patient_df.patient_id == elem]['tensor_indices_valid'].values
        elif subset_name == 'test':
            feature_indices = feature_df[feature_df.patient_id == elem]['tensor_indices_test'].values
            general_indices = patient_df[patient_df.patient_id == elem]['tensor_indices_test'].values
        t = feature_tensor[feature_indices]
        p = patient_tensor[general_indices]
        # transform
        indices_of_visits = torch.tensor([elem for elem in range(len(t)) if t[elem, visit_flag] == 1])
        
        if method == 'predict_all_visits':
            # predict all the visits after first visit
            flag = 1
        else :
            # predict only visits such that hist size is long enough (if possible, else impute)
            flag = -1
        # not enough rows
        if indices_of_visits[flag] < hist_size + 1:
            if elem == debug_patient:
                print(
                    f'history too short for debug patient, length of history before first visit to predict {indices_of_visits[flag]-1} minimum required length {hist_size} appending {hist_size + 1 - indices_of_visits[flag]} rows')
            t = torch.cat([torch.full(size=(hist_size + 1 - indices_of_visits[flag], t.shape[1]), fill_value=-1, device=device), t], axis=0)
            indices_of_visits += (hist_size + 1 - indices_of_visits[flag])
    
        visits_to_predict = torch.tensor([elem for index, elem in enumerate(indices_of_visits) if elem > hist_size and index > 0])
        tensor_x = torch.empty(size=(len(visits_to_predict), t.shape[1] * hist_size + p.shape[1] + 1), device=device)
        tensor_y = torch.empty(size=(len(visits_to_predict), 1), device =device)
        num_targets[index_in_subset] = len(visits_to_predict)
        for index, visit in enumerate(visits_to_predict):
            tensor_x[index] = torch.cat([t[visit - hist_size:visit].flatten(), p.flatten(), t[visit, visit_date_flag].reshape(1)])
            tensor_y[index] = t[visit, target_flag].reshape(1, 1)
        input_tensor = torch.cat([input_tensor, tensor_x])
        input_targets = torch.cat([input_targets, tensor_y])
        if elem == debug_patient:
            print(f'targets to predict : {tensor_y}')

    return input_tensor, input_targets, num_targets


def train_and_test_pipeline_baseline(dataset, batch_size, n_epochs, hist_size, debug_patient = False):
    # device
    #TODO implement debug patient for baseline
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'device {device}')
    #train
    loss_per_epoch, loss_per_epoch_valid, model = train_model(
        dataset, batch_size, n_epochs, hist_size, device, debug_patient=debug_patient)
    # #test
    with torch.no_grad():
        feature_df = dataset.joint_df_proc.copy()
        patient_df = dataset.patients_df_proc.copy()
        feature_tensor_test = dataset.joint_df_scaled_tensor_test.detach().clone().to(device)
        patient_tensor_test = dataset.patients_df_scaled_tensor_test.detach().clone().to(device)
        #subset = ['0257c010-933a-1a73-3b48-46a2c6b254e8', '07367bce-2706-3fe9-acc5-800560f87048']
        input_tensor_test, targets_test, num_targets_per_patient = get_input_tensors(device, 
            dataset.test_ids, 'test', feature_df, patient_df, feature_tensor_test, patient_tensor_test, hist_size, debug_patient=None)
        out_test = model(input_tensor_test)
        results_df = create_results_df(device, dataset.test_ids, dataset, out_test, algo = 'baseline', num_targets=num_targets_per_patient)



    if len(loss_per_epoch_valid) > 0:
        plt.plot(range(0, len(loss_per_epoch), 1), loss_per_epoch)
        plt.plot(range(0, len(loss_per_epoch), int(len(loss_per_epoch) / len(loss_per_epoch_valid))), loss_per_epoch_valid)
    return loss_per_epoch, loss_per_epoch_valid, results_df
