import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):

    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context):

        B, L, D = x.shape
        Hp = Wp = int(L ** 0.5)

        x_norm = self.norm1(x)
        context_norm = self.norm2(context)

        q = self.q(x_norm).reshape(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)          # [B, H, L, d]
        kv = self.kv(context_norm).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]                                                                           # [B, H, N, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale                                                # [B, H, L, N]
        attn_weights = attn.softmax(dim=-1)

        # Computing spatial attention maps
        attn_map = attn_weights.mean(1)              # [B, L, N]
        attn_map = attn_map.sum(-1)                  # [B, L]
        attn_map = attn_map.view(B, Hp, Wp)       # [B, Hp, Wp]

        out = (attn_weights @ v).transpose(1, 2).reshape(B, L, D)
        out = self.proj(out)

        return F.dropout(out, p=0.2, training=self.training) + x, attn_map