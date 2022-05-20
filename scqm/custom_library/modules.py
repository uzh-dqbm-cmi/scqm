# modules of adaptivenet
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


#TODO add l1-regularization for the weights of the network ? (inside of trainer)

class LSTMModule(nn.Module):
    """
    LSTM feature encoder
    """

    # def __init__(self, config: Dict):
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


class LSTMModulebis(nn.Module):
    """
    LSTM feature encoder
    """

    # def __init__(self, config: Dict):
    def __init__(self, input_size, device='cpu', batch_first=True, hidden_size=12, num_layers=1):
        super(LSTMModulebis, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=batch_first)

    def forward(self, x):
        # x is PackedSequence object
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, len(x),
                         self.hidden_size, device=self.device).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, len(x), self.hidden_size, device=self.device).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        return lstm_out, (hn, cn)

#note both encoders are actually the same.

class Encoder(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_features, size_out, num_hidden=2, hidden_size=10, p=0):
        super(Encoder, self).__init__()
        self.size_embedding = size_out
        self.input_layer = nn.Linear(num_features, hidden_size)
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


class PaddedEncoder(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_features, size_out, size_padding, num_hidden=2, hidden_size=10, p=0):
        super(PaddedEncoder, self).__init__()
        self.size_embedding = size_padding
        self.size_out = size_out
        self.input_layer = nn.Linear(num_features, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.out = nn.Linear(hidden_size, size_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return F.pad(self.out(x), pad =(0, self.size_embedding-self.size_out))


class PaddedEncoderBis(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_features, size_out, size_padding, num_hidden = 3, p=0):
        super(PaddedEncoderBis, self).__init__()
        self.size_embedding = size_padding
        self.size_out = size_out
        self.hidden_sizes = torch.tensor(
            [max(num_features - i * max(int((num_features - size_out) / (num_hidden + 1)), 1), size_out) for i in range(1, num_hidden + 1)])
        self.input_layer = nn.Linear(num_features, self.hidden_sizes[0])
        self.linears = nn.ModuleList([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]) for i in range(0, num_hidden-1)])
        self.out = nn.Linear(self.hidden_sizes[-1], size_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return F.pad(self.out(x), pad=(0, self.size_embedding - self.size_out))

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, num_hidden=2, hidden_size=10):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self,x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        return self.out(x)


class DecoderBis(nn.Module):
    def __init__(self, input_size, output_size, num_hidden=3):
        super(DecoderBis, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = torch.tensor(
            [max(input_size - i * max(int((input_size - output_size) / (num_hidden + 1)), 1), output_size) for i in range(1, num_hidden + 1)])
        self.input_layer = nn.Linear(input_size, self.hidden_sizes[0])
        self.linears = nn.ModuleList([nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
                                     for i in range(0, num_hidden - 1)])
        self.out = nn.Linear(self.hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        return self.out(x)

class HistoryDecoder(nn.Module):
    def __init__(self, history_size, event_size, max_num_events, num_hidden=2, hidden_size=10):
        super(HistoryDecoder, self).__init__()
        self.history_size = history_size
        self.event_size = event_size
        self.max_num_events = max_num_events
        self.input_layer = nn.Linear(history_size, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.out = nn.Linear(hidden_size, event_size*max_num_events)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        out = torch.reshape(self.out(x), shape=(len(x), self.max_num_events, self.event_size))
        return out 

class PredModule(nn.Module):
    """Module to perform final prediction"""

    def __init__(self, input_size, output_size, num_hidden=2, hidden_size = 10, p= 0):
        super(PredModule, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.dropout = nn.Dropout(p=p)
        self.out = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.out(x)  


class MLP(nn.Module):
    """
    MLP baseline
    """

    def __init__(self, input_size, output_size = 1,num_hidden=2, hidden_size=30):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)])
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        return self.out(x)


if __name__ == "__main__":
    # test pack/unpack
    seq = [torch.tensor([[1., 1., 1.], [1., 1., 1.]]), torch.tensor([[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]])]
    lengths = [2, 3]
    padded_seq = torch.nn.utils.rnn.pad_sequence(
        seq, batch_first=True)
    pack_padded_seq = torch.nn.utils.rnn.pack_padded_sequence(
        padded_seq, batch_first=True, lengths=lengths, enforce_sorted=False)
    unpack = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_seq, batch_first=True)[0]
    att = torch.nn.Parameter(torch.tensor([[1.],[2.],[3.]]), requires_grad=True)
    att_weights = torch.matmul(unpack, att)
    att_weights_soft = torch.nn.Softmax(dim=1)(att_weights)

    # test encoder
    encoder = Encoder(2, 5)
    decoder = Decoder(5, 2)
    padded_encoder = PaddedEncoder(2, 3, 5)
    tensor = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    t = padded_encoder(tensor)
    print(t)
    t_dec = decoder(t)
    print(t_dec)
    # test lstm
    n_data = 5
    input_size = 4
    hidden_size = 2
    num_layers = 2
    sequence = []
    lengths = []
    batch_first = True
    attention = True
    # attention vector
    att = torch.randn(size=(hidden_size, 1), requires_grad=True)
    for elem in range(n_data):
        length = random.randint(1, 6)
        sequence.append(torch.randn(length, input_size))
        lengths.append(length)
    padded_seq = torch.nn.utils.rnn.pad_sequence(
        sequence, batch_first=batch_first)
    pack_padded_seq = torch.nn.utils.rnn.pack_padded_sequence(
        padded_seq, batch_first=True, lengths=lengths, enforce_sorted=False)
    # instantiate LSTM
    lstm = LSTMModule(input_size, 'cpu', batch_first=True, hidden_size=hidden_size, num_layers=3)
    # self, history_size, event_size, max_num_events, num_hidden=2, hidden_size=10
    output, (hn, cn) = lstm(pack_padded_seq)
    # compute attention weights
    output_unp = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=batch_first)[0]
    att_weights = torch.matmul(output_unp, att)
    att_weights_soft = torch.nn.Softmax(dim=1)(att_weights)
    # test prediction module
    if attention:
        input = torch.sum(output_unp * att_weights_soft, dim=1)
    else:
        input = hn[-1]
    mask = torch.ones(n_data, max(lengths), input_size)
    for index, elem in enumerate(sequence):
        mask[index,lengths[index]:] = 0 

    pred_module = PredModule(hidden_size, 1)
    out = pred_module(input)

    encoder_bis = PaddedEncoderBis(13, 3, 5, 3)
    print(out)
