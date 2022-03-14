# modules of adaptivenet
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

#TODO add l1-regularization for the weights of the network ?

# phi visit/medication


class VisitEncoder(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_visit_features, size_out=10, hidden_1=10, hidden_2=10):
        super(VisitEncoder, self).__init__()
        self.fc1 = nn.Linear(num_visit_features, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, size_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class MedicationEncoder(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_medications_features, size_out=10, hidden_1=10, hidden_2=10):
        super(MedicationEncoder, self).__init__()
        self.fc1 = nn.Linear(num_medications_features, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, size_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class LSTMModule(nn.Module):
    """
    LSTM feature encoder
    """

    def __init__(self, input_size, batch_first = True, hidden_size=12):
        super(LSTMModule, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=batch_first)

    def forward(self, x):
        # x is PackedSequence object
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, max(x.batch_sizes).item(), self.hidden_size).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, max(
            x.batch_sizes).item(), self.hidden_size).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        return lstm_out, (hn, cn)


class PredModule(nn.Module):
    """Module to perform final prediction"""

    def __init__(self, input_size, hidden_1=10, hidden_2=10):
        super(PredModule, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.out = nn.Linear(hidden_2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class MLP(nn.Module):
    """
    MLP baseline
    """

    def __init__(self, input_size, hidden_1=10, hidden_2=10):
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
    lstm = LSTMModule(input_size, hidden_size)
    output, (hn, cn) = lstm(pack_padded_seq)
    # test prediction module
    input = hn[-1]
    input_size = hidden_size
    pred_module = PredModule(input_size)
    out = pred_module(input)