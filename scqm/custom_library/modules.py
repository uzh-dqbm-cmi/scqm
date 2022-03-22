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
    
    def set_training_parameters(self, n_epochs, batch_size, lr, min_num_visits, balance_classes):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.min_num_visits = min_num_visits
        self.balance_classes = balance_classes
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.lr)
        
        


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
