import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention_Light(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        B, L, D = x.shape
        H = W = int(L ** 0.5)

        q = self.q(x).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = attn.softmax(dim=-1)   # [B, heads, L, N]

        attn_map = attn_weights.mean(1)   # [B, heads, L, N] → [B, L, N]
        attn_map = attn_map.sum(-1)  # [B, L]
        attn_map = attn_map.view(B, H, W)

        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)
        return out + x, attn_map   # attn_map: [B, H, W]


class AttentionGuidedFusion_Light(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        mid_dim = max(dim // reduction, 8)

        self.text_to_vis = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        self.cov_net = nn.Sequential(
            nn.Conv2d(dim + 1, mid_dim, 1),
            nn.GELU(),
            nn.Conv2d(mid_dim, 3, 1)   # log_Q, log_R, gate_bias
        )

        self.norm = nn.LayerNorm(dim)
        self.sim_temp = nn.Parameter(torch.tensor(0.1))

    def forward(self, vis_feat, text_feat, attn_map):
        B, D, H, W = vis_feat.shape

        text_proj = self.text_to_vis(text_feat).mean(1)
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

        residual = text_proj - vis_feat
        sim = F.cosine_similarity(vis_feat, text_proj, dim=1) / self.sim_temp

        attn_map = attn_map.unsqueeze(1)

        concat = torch.cat([vis_feat, attn_map], dim=1)
        cov = self.cov_net(concat)
        log_Q, log_R, gate_bias = cov.chunk(3, dim=1)

        Q = torch.exp(log_Q) + 1e-6
        R = torch.exp(log_R) + 1e-6

        K_prior = torch.sigmoid(attn_map)
        K = 0.8 * K_prior + 0.2 * (Q / (Q + R + 1e-6))
        gate = torch.sigmoid(gate_bias + 4.0 * attn_map)

        delta = K * residual
        fused = vis_feat + gate * delta * sim.unsqueeze(1)

        fused = fused.flatten(2).permute(0, 2, 1)
        return self.norm(fused).permute(0, 2, 1).reshape(B, D, H, W)

# Boundary Branch
class BoundaryBranch(nn.Module):

    def __init__(self, in_channels, hidden_dim=64, patch_size=4, dropout_p=0.2):
        super().__init__()

        self.edge_extractor = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),

            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.text_proj = nn.Linear(768, hidden_dim)

        self.ape = AttentionGuidedFusion_Light(
            dim=hidden_dim,
            reduction=16
        )

        self.boundary_proj = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=patch_size, stride=patch_size)

        self.boundary_cross_attn = CrossAttention_Light(
            dim=hidden_dim, num_heads=4
        )


    def forward(self, feat, text_feat):
        # feat: [B, in_channels, H, W]  (c1, H/4)
        edge_feat = self.edge_extractor(feat)   # [B, hidden_dim, H, W]

        text_proj = self.text_proj(text_feat) # [B, N, hidden_dim]

        B, N, D = text_proj.shape
        H, W = edge_feat.shape[2:]

        edge_flat = edge_feat.flatten(2).permute(0, 2, 1)
        edge_flat = F.layer_norm(edge_flat, edge_flat.shape[-1:])

        edge_enhanced, boundary_attn_map = self.boundary_cross_attn(edge_flat, text_proj)

        edge_enhanced = edge_enhanced.permute(0, 2, 1).reshape(B, D, H, W)

        boundary_attn_map = F.interpolate(
            boundary_attn_map.unsqueeze(1),
            size=(H, W),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # [B, H, W]

        edge_updated = self.ape(edge_enhanced, text_proj, boundary_attn_map)

        boundary = self.boundary_proj(edge_updated)    # [B, 1, H, W]
        boundary = self.upsample(boundary)

        return boundary