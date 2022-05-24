import sys

sys.path.append("../scqm")
import torch
import random
from scqm.custom_library.models.modules.encoders import EventEncoder
from scqm.custom_library.models.modules.lstms import LstmEventSpecific
from scqm.custom_library.models.modules.predictions import PredModule

if __name__ == "__main__":
    batch_first = True
    device = "cpu"
    names = ["visit", "med"]
    config = {
        "visit": {"num_features": 3, "size_out": 2, "size_history": 2},
        "med": {"num_features": 4, "size_out": 3, "size_history": 2},
    }
    encoders = {
        name: EventEncoder(config[name]["num_features"], config[name]["size_out"])
        for name in names
    }
    lstms = {
        name: LstmEventSpecific(
            config[name]["size_out"], device, batch_first, config[name]["size_history"]
        )
        for name in names
    }
    pred = PredModule(sum([config[name]["size_history"] for name in names]), 1)

    # test lstms
    n_data = 2
    max_seq_length = 5
    combined = {
        name: torch.zeros(
            size=(
                n_data,
                max_seq_length,
                config[name]["size_out"],
            ),
            device=device,
        )
        for name in names
    }
    lstm_out = {}
    for name in names:
        input_size = config[name]["num_features"]
        # attention vector
        att = torch.randn(size=(config[name]["size_history"], 1), requires_grad=True)
        lengths = []
        for elem in range(n_data):
            length = random.randint(1, max_seq_length - 1)
            lengths.append(length)
            values = torch.randn(length, input_size)
            combined[name][elem, 0:length, :] = encoders[name](values)
        # apply lstm
        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(
            combined[name], lengths, enforce_sorted=False, batch_first=batch_first
        )
        output, (hn, cn) = lstms[name](packed_sequence)
        # compute unpacked output
        unpacked_output = torch.nn.utils.rnn.pad_packed_sequence(
            output, batch_first=batch_first
        )[0]
        attention_weights = torch.nn.Softmax(dim=1)(torch.matmul(unpacked_output, att))
    print("End of script")
