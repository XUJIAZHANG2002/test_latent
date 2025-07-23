import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepUNet(nn.Module):
    def __init__(self, in_channels=4, T=500):
        super(DeepUNet, self).__init__()
        self.T = T

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Embedding(T, 256), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 512)
        )

        # Downsampling
        self.down1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.down1_bn = nn.BatchNorm2d(32)
        self.down2 = nn.Conv2d(32, 64, 3, padding=1)
        self.down2_bn = nn.BatchNorm2d(64)
        self.down3 = nn.Conv2d(64, 128, 3, padding=1)
        self.down3_bn = nn.BatchNorm2d(128)
        self.down4 = nn.Conv2d(128, 256, 3, padding=1)
        self.down4_bn = nn.BatchNorm2d(256)
        self.down5 = nn.Conv2d(256, 512, 3, padding=1)
        self.down5_bn = nn.BatchNorm2d(512)

        # Bottleneck
        self.bottleneck = nn.Conv2d(512, 512, 3, padding=1)
        self.bottleneck_bn = nn.BatchNorm2d(512)

        # Upsampling
        self.up1 = nn.Conv2d(512 + 256, 256, 3, padding=1)
        self.up1_bn = nn.BatchNorm2d(256)
        self.up2 = nn.Conv2d(256 + 128, 128, 3, padding=1)
        self.up2_bn = nn.BatchNorm2d(128)
        self.up3 = nn.Conv2d(128 + 64, 64, 3, padding=1)
        self.up3_bn = nn.BatchNorm2d(64)
        self.up4 = nn.Conv2d(64 + 32, 32, 3, padding=1)
        self.up4_bn = nn.BatchNorm2d(32)
        self.up5 = nn.Conv2d(32 + in_channels, 16, 3, padding=1)
        self.up5_bn = nn.BatchNorm2d(16)

        # Output layer
        self.out = nn.Conv2d(16, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_emb(t).view(-1, 512, 1, 1).repeat(1, 1, x.size(2), x.size(3))

        # Downsampling
        d1 = F.relu(self.down1_bn(self.down1(x)))
        d2 = F.relu(self.down2_bn(self.down2(d1)))
        d3 = F.relu(self.down3_bn(self.down3(d2)))
        d4 = F.relu(self.down4_bn(self.down4(d3)))
        d5 = F.relu(self.down5_bn(self.down5(d4)))

        # Bottleneck
        b = F.relu(self.bottleneck_bn(self.bottleneck(d5)) + t_emb)

        # Upsampling
        u1 = F.relu(self.up1_bn(self.up1(torch.cat([b, d4], dim=1))))
        u2 = F.relu(self.up2_bn(self.up2(torch.cat([u1, d3], dim=1))))
        u3 = F.relu(self.up3_bn(self.up3(torch.cat([u2, d2], dim=1))))
        u4 = F.relu(self.up4_bn(self.up4(torch.cat([u3, d1], dim=1))))
        u5 = F.relu(self.up5_bn(self.up5(torch.cat([u4, x], dim=1))))

        return self.out(u5)
