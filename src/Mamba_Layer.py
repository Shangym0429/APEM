import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class MambaBlock(nn.Module):

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + F.dropout(self.mamba(self.norm(x)), p=0.2, training=self.training)