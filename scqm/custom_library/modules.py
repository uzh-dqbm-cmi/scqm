# modules of adaptivenet
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO add l1-regularization for the weights of the network ?

class LSTMModule(nn.Module):
    """
    LSTM feature encoder
    """

    def __init__(self, input_size, device='cpu', batch_first=True, hidden_size=12, num_layers = 1):
        super(LSTMModule, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=batch_first)

    def forward(self, x):
        # x is PackedSequence object
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, max(x.batch_sizes).item(), self.hidden_size, device = self.device).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, max(
            x.batch_sizes).item(), self.hidden_size, device = self.device).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        return lstm_out, (hn, cn)


class VisitEncoder(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_visit_features, size_out, num_hidden=2, hidden_size=10, p=0):
        super(VisitEncoder, self).__init__()
        self.size_embedding = size_out
        self.input_layer = nn.Linear(num_visit_features, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.out = nn.Linear(hidden_size, size_out)
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.out(x)


class MedicationEncoder(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_medications_features, size_out, num_hidden=2, hidden_size=10, p=0):
        super(MedicationEncoder, self).__init__()
        self.input_layer = nn.Linear(num_medications_features, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.out = nn.Linear(hidden_size, size_out)
        self.dropout = nn.Dropout(p)
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.out(x)



class PredModule(nn.Module):
    """Module to perform final prediction"""

    def __init__(self, input_size, num_hidden=2, hidden_size = 10, p= 0):
        super(PredModule, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.dropout = nn.Dropout(p=p)
        self.out = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.out(x)


class Model:
    def __init__(self, model_specifics, device):
        self.device = device
        self.VEncoder = VisitEncoder(model_specifics['num_visit_features'], model_specifics['size_embedding'], model_specifics['num_layers_enc'],
                                model_specifics['hidden_enc'], model_specifics['dropout']).to(device)
        self.MEncoder = MedicationEncoder(model_specifics['num_medications_features'], model_specifics['size_embedding'],
                                 model_specifics['num_layers_enc'], model_specifics['hidden_enc'], model_specifics['dropout']).to(device)
        self.LModule = LSTMModule(model_specifics['size_embedding'], model_specifics['device'],
                         model_specifics['batch_first'], model_specifics['size_history'], model_specifics['num_layers']).to(device)
        # + 1 for time to prediction
        self.PModule = PredModule(model_specifics['size_history'] + model_specifics['num_general_features'] +
                         1, model_specifics['num_layers_pred'], model_specifics['hidden_pred'], model_specifics['dropout']).to(device)
        self.task = model_specifics['task']
        self.batch_first = model_specifics['batch_first']
        self.parameters = list(self.VEncoder.parameters()) + list(self.MEncoder.parameters()) + \
        list(self.LModule.parameters()) + list(self.PModule.parameters())
    
    def train(self):
        self.VEncoder.train()
        self.MEncoder.train()
        self.LModule.train()
        self.PModule.train()
    
    def eval(self):
        self.VEncoder.eval()
        self.MEncoder.eval()
        self.LModule.eval()
        self.PModule.eval()

    def apply_and_get_loss(self, dataset, criterion, batch, metrics):
        loss = 0
        o_v, o_m = self.VEncoder(dataset.visits_df_scaled_tensor_train[batch.indices_v]), self.MEncoder(
            dataset.medications_df_scaled_tensor_train[batch.indices_m])
        # for scaling of loss
        num_targets = 0
        for v in range(0, batch.max_num_visits - dataset.min_num_visits + 1):
            # stores for all the patients in the batch the tensor of ordered events (of varying size)
            sequence = []
            # to keep track of the right index in the visits/medication tensors
            index_visits = 0
            index_medications = 0
            # targets (values)
            target_values = torch.empty(size=(torch.sum(batch.visit_mask[:, v] == True).item(), 1), device=self.device)
            # targets (categories : 0 if <= 2.6 else 1)
            target_categories = torch.empty(size=(torch.sum(batch.visit_mask[:, v] == True).item(), 1), device=self.device)
            # delta t
            time_to_targets = torch.empty(size=(torch.sum(batch.visit_mask[:, v] == True).item(), 1), device=self.device)
            # for each patient combine the medication and visit events in the right order up to visit v
            index_target = 0
            debug_index_target = None
            for patient, seq in enumerate(batch.seq_lengths[v]):
                # check if the patient has at least v visits
                if batch.visit_mask[patient, v] == True:
                    # create combined ordered list of visit/medication events up to v
                    combined = torch.zeros_like(torch.cat([o_v[index_visits:index_visits + seq[0]],
                                                        o_m[index_medications:index_medications + seq[1]]]))
                    combined[batch.masks[patient][v]] = o_v[index_visits:index_visits + seq[0]].flatten()
                    combined[~batch.masks[patient][v]] = o_m[index_medications:index_medications + seq[1]].flatten()

                    sequence.append(combined)
                    target_values[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_t][index_visits + seq[0], 1]
                    time_to_targets[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_t][index_visits + seq[0], 0]
                    target_categories[index_target] = dataset.targets_df_scaled_tensor_train[batch.indices_t][index_visits + seq[0], 2]
                    if batch.debug_index != None and patient == batch.debug_index:
                        print(f'next target value: {target_values[index_target]}')
                        print(f'Next target category : {target_categories[index_target]}')
                        debug_index_target = index_target
                        #print(f'visit info {tensor_v[index_visits:index_visits + seq[0]]} \n medication info {tensor_m[index_medications:index_medications + seq[1]]}')
                    index_target += 1
                # update the indices to select from in the tensors
                index_visits += batch.total_num[patient, 0]
                index_medications += batch.total_num[patient, 1]
            # "preprocessing" to apply lstm
            padded_sequence = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=self.batch_first)
            # compute the lengths of the sequences for each patient with available visit v
            lengths = batch.seq_lengths[v].sum(dim=1)[batch.visit_mask[:, v]]

            pack_padded_sequence = torch.nn.utils.rnn.pack_padded_sequence(
                padded_sequence, batch_first=self.batch_first, lengths=lengths, enforce_sorted=False)

            # apply lstm
            output, (hn, cn) = self.LModule(pack_padded_sequence)
            history = hn[-1]
            # concat computed patient history with general information
            pred_input = torch.cat(
                (dataset.patients_df_scaled_tensor_train[batch.indices_p][batch.visit_mask[:, v]], history, time_to_targets), dim=1)
            # apply prediction module
            out = self.PModule(pred_input)
            # compute loss
            if self.task == 'classification':
                targets = target_categories
            else:
                targets = target_values
            loss += criterion(out, targets)
            num_targets += len(targets)
            # compute other training metrics
            if self.task == 'classification':
                predictions = torch.tensor([1 if elem >= 0.5 else 0 for elem in out], device=self.device)
            else:
                predictions = out
            metrics.add_observations(predictions, targets)

            if debug_index_target != None:
                # print(out)
                print(f'prediction {out[debug_index_target].item()}')

        return loss / num_targets, metrics
    


class MLP(nn.Module):
    """
    MLP baseline
    """

    def __init__(self, input_size, hidden_1=30, hidden_2=30):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


if __name__ == "__main__":
    # test encoder
    encoder = VisitEncoder(2, 5)
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    encoder(tensor)
    # test lstm
    n_data = 5
    input_size = 4
    hidden_size = 10
    num_layers = 1
    sequence = []
    lengths = []
    batch_first = True
    for elem in range(n_data):
        length = random.randint(1, 6)
        sequence.append(torch.randn(length, input_size))
        lengths.append(length)
    padded_seq = torch.nn.utils.rnn.pad_sequence(
        sequence, batch_first=batch_first)
    pack_padded_seq = torch.nn.utils.rnn.pack_padded_sequence(
        padded_seq, batch_first=True, lengths=lengths, enforce_sorted=False)
    # instanciate LSTM
    lstm = LSTMModule(input_size, 'cpu', batch_first=True, hidden_size=hidden_size, num_layers=1)
    output, (hn, cn) = lstm(pack_padded_seq)
    # test prediction module
    input = hn[-1]
    input_size = hidden_size
    pred_module = PredModule(input_size)
    out = pred_module(input)
    print(out)
