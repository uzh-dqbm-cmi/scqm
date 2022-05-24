import torch.nn as nn
import torch.nn.functional as F


class PredModule(nn.Module):
    """Module to perform final prediction"""

    def __init__(self, input_size, output_size, num_hidden=2, hidden_size=10, p=0):
        super(PredModule, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)]
        )
        self.dropout = nn.Dropout(p=p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        for l in self.linears:
            x = F.relu(l(x))
            x = self.dropout(x)
        return self.out(x)
