import torch.nn as nn
import torch.nn.functional as F
import torch


class EventDecoder(nn.Module):
    """
    Mapps the embedding to initial features
    """

    def __init__(self, input_size, output_size, num_hidden=2, hidden_size=10):
        super(EventDecoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)]
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        return self.out(x)


class EventDecoderFixed(nn.Module):
    """
    Decoder with hidden sizes with deterministic decreasing number of neurons
    """

    def __init__(self, input_size, output_size, num_hidden=3):
        super(EventDecoderFixed, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = torch.tensor(
            [
                max(
                    input_size
                    - i * max(int((input_size - output_size) / (num_hidden + 1)), 1),
                    output_size,
                )
                for i in range(1, num_hidden + 1)
            ]
        )
        self.input_layer = nn.Linear(input_size, self.hidden_sizes[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
                for i in range(0, num_hidden - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_sizes[-1], output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        return self.out(x)
