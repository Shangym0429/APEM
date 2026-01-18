import torch.nn as nn

class APEM(nn.Module):
    def __init__(self, bert_model_name='agriculture-bert-uncased', num_classes=1, patch_size=4,
                 max_text_tokens=max_text_tokens):
        super().__init__()

        self.vision_encoder = VisionEncoder()
        self.text_encoder = TextEncoder(bert_model_name, max_text_tokens=max_text_tokens)

        self.decoder4 = MultimodalFusionDecoder(1024, 768, hidden_dim=512, num_mamba_layers=4, num_attn_layers=2,
                                                patch_size=patch_size)
        self.decoder3 = MultimodalFusionDecoder(512, 768, hidden_dim=256, num_mamba_layers=3, num_attn_layers=2,
                                                patch_size=patch_size)
        self.decoder2 = MultimodalFusionDecoder(256, 768, hidden_dim=128, num_mamba_layers=2, num_attn_layers=1,
                                                patch_size=patch_size)
        self.decoder1 = MultimodalFusionDecoder(128, 768, hidden_dim=64, num_mamba_layers=1, num_attn_layers=1,
                                                patch_size=patch_size)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.up1_1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.up1_2 = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.seg_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )

        # c1
        self.boundary_branch = BoundaryBranch(128, hidden_dim=64, patch_size=4)

    def forward(self, image, input_ids, attention_mask):
        c1, c2, c3, c4 = self.vision_encoder(image)
        text_global, text_seq = self.text_encoder(input_ids, attention_mask)

        d4, align4 = self.decoder4(c4, text_seq)
        d3, align3 = self.decoder3(c3, text_seq)
        d2, align2 = self.decoder2(c2, text_seq)
        d1, align1 = self.decoder1(c1, text_seq)

        align_loss = (align1 + align2 + align3 + align4) / 4.0

        d3 = self.up4(d4)
        d3 = d3 + self.decoder3(c3, text_seq)[0]

        d2 = self.up3(d3)
        d2 = d2 + self.decoder2(c2, text_seq)[0]

        d1 = self.up2(d2)
        d1 = d1 + self.decoder1(c1, text_seq)[0]

        d0 = self.up1_1(d1)
        d0 = self.up1_2(d0)

        boundary_map = self.boundary_branch(c1, text_seq)
        out = self.seg_head(d0) + 0.3 * boundary_map

        return out, align_loss