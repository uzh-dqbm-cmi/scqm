import torch
import torch.nn as nn
import torch.nn.functional as F


class EventEncoder(nn.Module):
    """
    Event encoder to fixed size embedding
    """

    def __init__(self, num_features, size_out, num_hidden=2, hidden_size=10, p=0):
        super(EventEncoder, self).__init__()
        self.size_embedding = size_out
        self.input_layer = nn.Linear(num_features, hidden_size)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)]
        )
        self.out = nn.Linear(hidden_size, size_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.out(x)


class PaddedEventEncoder(nn.Module):
    """
    Padded encoder. Padds the output to a fixed size size_out
    """

    def __init__(
        self, num_features, size_out, size_embedding, num_hidden=2, hidden_size=10, p=0
    ):
        super(PaddedEventEncoder, self).__init__()
        self.size_embedding = size_embedding
        self.size_out = size_out
        self.input_layer = nn.Linear(num_features, hidden_size)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)]
        )
        self.out = nn.Linear(hidden_size, size_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return F.pad(self.out(x), pad=(0, self.size_embedding - self.size_out))


class PaddedEncoderFixed(nn.Module):
    """
    visit encoder
    """

    def __init__(self, num_features, size_out, size_padding, num_hidden=3, p=0):
        super(PaddedEncoderFixed, self).__init__()
        self.size_embedding = size_padding
        self.size_out = size_out
        self.hidden_sizes = torch.tensor(
            [
                max(
                    num_features
                    - i * max(int((num_features - size_out) / (num_hidden + 1)), 1),
                    size_out,
                )
                for i in range(1, num_hidden + 1)
            ]
        )
        self.input_layer = nn.Linear(num_features, self.hidden_sizes[0])
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
                for i in range(0, num_hidden - 1)
            ]
        )
        self.out = nn.Linear(self.hidden_sizes[-1], size_out)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return F.pad(self.out(x), pad=(0, self.size_embedding - self.size_out))
