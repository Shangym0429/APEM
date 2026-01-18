import os

import torch
import torch.nn as nn
from torchgeo.models import swin_v2_b


class VisionEncoder(nn.Module):
    def __init__(self, local_weight_path=None):
        super().__init__()
        if local_weight_path is None:
            local_weight_path = "swin_v2_b_naip_rgb_satlas"


        self.swin = swin_v2_b(weights=None)

        if os.path.exists(local_weight_path):
            state_dict = torch.load(local_weight_path, map_location='cpu', weights_only=True)

        # Freezing shallow networks
        for name, param in self.swin.named_parameters():
            if any(x in name for x in ['features.0', 'features.1', 'features.2', 'features.3']):
                param.requires_grad = False

    def forward(self, x):

        out = self.swin.features(x)
        out = out.permute(0, 3, 1, 2)

        x = self.swin.features[0](x)
        c1 = self.swin.features[1](x)   # [B, H/4,  W/4,  128]
        x  = self.swin.features[2](c1)
        c2 = self.swin.features[3](x)   # [B, H/8,  W/8,  256]
        x  = self.swin.features[4](c2)
        c3 = self.swin.features[5](x)   # [B, H/16, W/16, 512]
        x  = self.swin.features[6](c3)
        c4 = self.swin.features[7](x)   # [B, H/32, W/32, 1024]

        c1 = c1.permute(0, 3, 1, 2)  # [B, 128, H/4,  W/4]
        c2 = c2.permute(0, 3, 1, 2)  # [B, 256, H/8,  W/8]
        c3 = c3.permute(0, 3, 1, 2)  # [B, 512, H/16, W/16]
        c4 = c4.permute(0, 3, 1, 2)  # [B, 1024, H/32, W/32]

        return c1, c2, c3, c4