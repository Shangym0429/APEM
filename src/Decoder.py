import torch
import torch.nn as nn
import torch.nn.functional as F

class MultimodalFusionDecoder(nn.Module):

    def __init__(self, img_channels, text_dim, hidden_dim=512, num_mamba_layers=1, num_attn_layers=1, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_attn_layers = num_attn_layers

        self.img_proj = nn.Conv2d(img_channels, hidden_dim, 1)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        self.cpe = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
        )

        self.patch_conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model=hidden_dim) for _ in range(num_mamba_layers)
        ])

        self.local_attns = nn.ModuleList([
            LocalSelfAttention(hidden_dim, num_heads=8) for _ in range(num_attn_layers)
        ])
        self.cross_attns = nn.ModuleList([
            CrossAttention(hidden_dim, num_heads=8) for _ in range(num_attn_layers)
        ])

        self.prior_enhanced = nn.ModuleList([
            AttentionGuidedFusion(hidden_dim) for _ in range(num_attn_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.upsample = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)

        self.register_buffer('align_temperature', torch.tensor(0.07))

    def compute_alignment_loss(self, img_feat_flat, text_feat_proj):
        img_emb = F.normalize(img_feat_flat.mean(dim=1), dim=-1)
        txt_emb = F.normalize(text_feat_proj.mean(dim=1), dim=-1)
        logits = img_emb @ txt_emb.t() / self.align_temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2

    def forward(self, img_feat, text_feat):
        B, C, H, W = img_feat.shape

        img_feat = self.img_proj(img_feat)
        img_feat = img_feat + self.cpe(img_feat)

        img_feat_patch = self.patch_conv(img_feat)
        Hp, Wp = img_feat_patch.shape[2], img_feat_patch.shape[3]
        img_feat_flat = img_feat_patch.flatten(2).permute(0, 2, 1)

        text_feat_proj = self.text_proj(text_feat)

        align_loss = self.compute_alignment_loss(img_feat_flat.detach(), text_feat_proj.detach())

        x = img_feat_flat
        attn_idx = 0
        insert_positions = [len(self.mamba_layers) // (self.num_attn_layers + 1) * (k + 1) - 1
                            for k in range(self.num_attn_layers)]

        for i, mamba_layer in enumerate(self.mamba_layers):
            x = mamba_layer(x)

            if attn_idx < self.num_attn_layers and i == insert_positions[attn_idx]:
                x = self.local_attns[attn_idx](x)

                x, attn_map = self.cross_attns[attn_idx](x, text_feat_proj)

                x_2d = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                x_2d = self.prior_enhanced[attn_idx](x_2d, text_feat_proj, attn_map)

                x = x_2d.flatten(2).permute(0, 2, 1)
                attn_idx += 1

        x = self.norm(x)
        x = F.dropout(x, p=0.1, training=self.training)

        x = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp)
        x = self.upsample(x)

        return x, align_loss