import torch
import torch.nn as nn
import torch.nn.functional as F


class LstmAllHistory(nn.Module):
    """
    LSTM feature encoder
    """

    # def __init__(self, config: Dict):
    def __init__(
        self, input_size, device="cpu", batch_first=True, hidden_size=12, num_layers=1
    ):
        super(LstmAllHistory, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
        )

    def forward(self, x):
        # x is PackedSequence object
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        return lstm_out, (hn, cn)


class LstmEventSpecific(nn.Module):
    """
    LSTM feature encoder
    """

    # def __init__(self, config: Dict):
    def __init__(
        self, input_size, device="cpu", batch_first=True, hidden_size=12, num_layers=1
    ):
        super(LstmEventSpecific, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
        )

    # def forward(self, x):
    #     # x is PackedSequence object
    #     # Initialize hidden state with zeros
    #     h0 = torch.zeros(
    #         self.num_layers, len(x), self.hidden_size, device=self.device
    #     ).requires_grad_()
    #     # Initialize cell state
    #     c0 = torch.zeros(
    #         self.num_layers, len(x), self.hidden_size, device=self.device
    #     ).requires_grad_()
    #     lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
    #     return lstm_out, (hn, cn)
    def forward(self, x):
        # x is PackedSequence object
        # Initialize hidden state with zeros
        h0 = torch.zeros(
            self.num_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(
            self.num_layers,
            max(x.batch_sizes).item(),
            self.hidden_size,
            device=self.device,
        ).requires_grad_()
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        return lstm_out, (hn, cn)
