import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    MLP baseline
    """

    def __init__(self, input_size:int, output_size:int=1, num_hidden:int=2, hidden_size:int=30):
        super(MLP, self).__init__()
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