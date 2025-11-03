from torch import nn
from utils.common import VAE_AttentionBlock, VAE_ResidualBlock

import torch.nn as nn
import torch

import torch.nn as nn

class HsiEncoder(nn.Module):
    def __init__(self, G_shape, out_channels):
        """
        G_shape: (C, H, W) 입력 채널 정보 중 C만 쓰임
        out_channels: μ, logvar 출력 채널 수
        """
        super().__init__()
        c = G_shape[0]
        M1 = 1
        M2 = 2
        M3 = 4

        self.init_block = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            VAE_ResidualBlock(c, c),
            VAE_AttentionBlock(c)
        )

        self.shared = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            VAE_ResidualBlock(c, c),
            VAE_AttentionBlock(c),
            VAE_ResidualBlock(c, c),
            VAE_AttentionBlock(c),

            nn.Conv2d(c, c*M1, kernel_size=3, padding=1),
            VAE_ResidualBlock(c*M1, c*M1),
            VAE_AttentionBlock(c*M1),
            VAE_ResidualBlock(c*M1, c*M1),
            VAE_AttentionBlock(c*M1),

            nn.Conv2d(c*M1, c*M2, kernel_size=3, padding=1),
            VAE_ResidualBlock(c*M2, c*M2),
            VAE_AttentionBlock(c*M2),
            VAE_ResidualBlock(c*M2, c*M2),
            VAE_AttentionBlock(c*M2),

            nn.SiLU(),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(c*M2, c*M3, kernel_size=3, stride=2, padding=1),
            VAE_ResidualBlock(c*M3, c*M3),
            VAE_AttentionBlock(c*M3),
            VAE_ResidualBlock(c*M3, c*M3),
            VAE_AttentionBlock(c*M3),
            nn.SiLU(),
        )
        self.out_conv = nn.Conv2d(c*M3, out_channels, kernel_size=1)


    def forward(self, x):
        h = self.init_block(x)         # (B, c,   H,   W)
        h = self.shared(h)             # (B, c*4, H/2, W/2)
        h = self.downsample(h)         # (B, c*8, H/4, W/4)

        return self.out_conv(h)



class RgbSegEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1), # (B, 4, 256, 256) -> (B, 8, 128, 128)
            VAE_ResidualBlock(8, 8), 
            VAE_AttentionBlock(8), 
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), # (B, 8, 128, 128) -> (B, 16, 64, 64)
            VAE_ResidualBlock(16, 16), 
            VAE_AttentionBlock(16), 
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # (B, 16, 64, 64) -> (B, 32, 32, 32)
            VAE_ResidualBlock(32, 32), 
            VAE_AttentionBlock(32), 
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # (B, 32, 32, 32) -> (B, 64, 16, 16)
            VAE_ResidualBlock(64, 64), 
            VAE_AttentionBlock(64), 
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # (B, 64, 16, 16) -> (B, 64, 16, 16)
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # (B, 64, 16, 16) -> (B, 64, 16, 16)
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x
