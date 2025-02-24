import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureToPseudoRGB(nn.Module):
    def __init__(self, hidden_dim=512, out_channels=3):
        super().__init__()
        # 调整全连接层输出维度为 64x48 (假设ViT的patch_size=16)
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 8 * 6 * 256)  # 输出为 8x6x256 (假设后续上采样到256x192)
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8x6 → 16x12
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16x12 → 32x24
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32x24 → 64x48
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64x48 → 128x96
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 128x96 → 256x192
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 1, 3, 1, 1)
        )

    def forward(self, features):
        print(features.shape)
        x = self.fc_layers(features)
        print(x.shape)
        x = x.view(-1, 256, 8, 6)
        print(x.shape)
        x = self.decoder(x)
        x = torch.cat([x, x, x], dim=1)
        assert x.shape[-2:] == (256, 192), f"输出分辨率错误，应为256x192，实际为{x.shape[-2:]}"
        return x
    

if __name__ == "__main__":
    model = FeatureToPseudoRGB()
    print(model)
    inputs = torch.randn(32, 512)
    outputs = model(inputs)
