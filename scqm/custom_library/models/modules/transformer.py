import torch
import torch.nn as nn
import torch.nn.functional as F
from scqm.custom_library.utils import SaveOutput


class TransformerSCQM(nn.Module):
    """
    Transformer
    """

    # def __init__(self, config: Dict):
    def __init__(
        self,
        dim_val: int,
        device: str = "cpu",
        batch_first: bool = True,
        n_heads: int = 4,
        dropout: float = 0.0,
        dim_feedforward_encoder: int = 100,
        n_encoder_layers: int = 2,
    ):
        super(TransformerSCQM, self).__init__()
        self.device = device
        self.input_size = dim_val
        self.n_heads = n_heads
        self.dropout = dropout

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=n_encoder_layers, norm=None
        )

    def forward(self, x, mask):

        return self.encoder(x, src_key_padding_mask=mask)
