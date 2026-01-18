import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGuidedFusion(nn.Module):

    def __init__(self, dim, reduction=8):
        super().__init__()
        self.dim = dim
        mid_dim = max(dim // reduction, 16)

        # language → vision
        self.text_to_vis = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # log_Q, log_R, gate_bias
        self.cov_net = nn.Sequential(
            nn.Conv2d(dim + 1, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, mid_dim, kernel_size=1),
            nn.BatchNorm2d(mid_dim),
            nn.GELU(),
            nn.Conv2d(mid_dim, 3, kernel_size=1)
        )

        self.norm = nn.LayerNorm(dim)
        self.sim_temp = nn.Parameter(torch.tensor(0.07))

    def forward(self, vis_feat, text_feat, attn_map=None):
        """
        vis_feat : [B, D, H, W]
        text_feat: [B, N, D]
        attn_map : [B, H, W]   comes from cross-attention
        """
        B, D, H, W = vis_feat.shape

        text_proj = self.text_to_vis(text_feat)          # [B, N, D]
        text_proj = text_proj.mean(dim=1)                 # [B, D]
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

        residual = text_proj - vis_feat

        sim = F.cosine_similarity(vis_feat, text_proj, dim=1, eps=1e-6) / self.sim_temp  # [B, H, W]

        # Attention Prior
        if attn_map is None:
            attn_map = sim.detach()

        attn_map = attn_map.unsqueeze(1)   # [B, 1, H, W]

        # 4. uncertainty parameters
        concat = torch.cat([vis_feat, attn_map], dim=1)    # [B, D+1, H, W]
        cov = self.cov_net(concat)                         # [B, 3, H, W]
        log_Q, log_R, gate_bias = cov.chunk(3, dim=1)

        Q = torch.exp(log_Q) + 1e-6
        R = torch.exp(log_R) + 1e-6


        K_prior = torch.sigmoid(attn_map)
        K_dynamic = Q / (Q + R + 1e-6)

        w = torch.softmax(self.fusion_weight, dim=0)
        K = w[0] * K_prior + w[1] * K_dynamic


        gate = torch.sigmoid(gate_bias + 3.0 * attn_map)

        # fusion
        delta = K * residual
        vis_fused = vis_feat + gate * delta * sim.unsqueeze(1)

        # Norm
        vis_fused = vis_fused.flatten(2).permute(0, 2, 1)  # [B, L, D]
        return self.norm(vis_fused).permute(0, 2, 1).reshape(B, D, H, W)
