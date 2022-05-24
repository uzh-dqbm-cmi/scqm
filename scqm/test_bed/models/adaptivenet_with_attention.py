import sys

sys.path.append("../scqm")
import torch
import random
from scqm.custom_library.models.modules.encoders import EventEncoder, PaddedEventEncoder
from scqm.custom_library.models.modules.decoders import EventDecoder
from scqm.custom_library.models.modules.lstms import LstmAllHistory
from scqm.custom_library.models.modules.predictions import PredModule

if __name__ == "__main__":
    # test pack/unpack
    seq = [
        torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
        torch.tensor([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
    ]
    lengths = [2, 3]
    padded_seq = torch.nn.utils.rnn.pad_sequence(seq, batch_first=True)
    pack_padded_seq = torch.nn.utils.rnn.pack_padded_sequence(
        padded_seq, batch_first=True, lengths=lengths, enforce_sorted=False
    )
    unpack = torch.nn.utils.rnn.pad_packed_sequence(pack_padded_seq, batch_first=True)[
        0
    ]
    att = torch.nn.Parameter(torch.tensor([[1.0], [2.0], [3.0]]), requires_grad=True)
    att_weights = torch.matmul(unpack, att)
    att_weights_soft = torch.nn.Softmax(dim=1)(att_weights)

    # test encoder
    encoder = EventEncoder(2, 5)
    decoder = EventDecoder(5, 2)

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
    padded_seq = torch.nn.utils.rnn.pad_sequence(sequence, batch_first=batch_first)
    pack_padded_seq = torch.nn.utils.rnn.pack_padded_sequence(
        padded_seq, batch_first=True, lengths=lengths, enforce_sorted=False
    )
    # instantiate LSTM
    lstm = LstmAllHistory(
        input_size, "cpu", batch_first=True, hidden_size=hidden_size, num_layers=3
    )
    # self, history_size, event_size, max_num_events, num_hidden=2, hidden_size=10
    output, (hn, cn) = lstm(pack_padded_seq)
    # compute attention weights
    output_unp = torch.nn.utils.rnn.pad_packed_sequence(
        output, batch_first=batch_first
    )[0]
    att_weights = torch.matmul(output_unp, att)
    att_weights_soft = torch.nn.Softmax(dim=1)(att_weights)
    # test prediction module
    if attention:
        input = torch.sum(output_unp * att_weights_soft, dim=1)
    else:
        input = hn[-1]
    mask = torch.ones(n_data, max(lengths), input_size)
    for index, elem in enumerate(sequence):
        mask[index, lengths[index] :] = 0

    pred_module = PredModule(hidden_size, 1)
    out = pred_module(input)
    print(out)
