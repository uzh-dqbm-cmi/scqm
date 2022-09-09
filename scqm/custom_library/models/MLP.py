from scqm.custom_library.models.model import Model
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(self, config, device, num_hidden: int = 2, hidden_size: int = 10):
        super(MLP, self).__init__()
        self.config = config
        self.device = device
        input_size = config["input_size"]
        output_size = config["output_size"]
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(1, num_hidden)]
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.to(device)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        for l in self.linears:
            x = F.relu(l(x))
        return self.out(x)
