from torch import nn
from torch.nn import functional as F
from utils.common import VAE_AttentionBlock, VAE_ResidualBlock

class HsiDecoder(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        c = in_channels
        o = out_channels

        max_heads = 16
        n_head = min(c, max_heads)

        while n_head > 1 and (c % n_head != 0 or o % n_head != 0):
            n_head //= 2

        min_ch, max_ch = sorted([c, o])
        mid = (min_ch + max_ch) // 2

        inter_mult = mid // n_head
        if inter_mult < 1:
            inter_mult = 1

        inter_channels = inter_mult * n_head

        super().__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            VAE_ResidualBlock(in_channels, in_channels, norm_channels=n_head),
            VAE_ResidualBlock(in_channels, in_channels, norm_channels=n_head),
            VAE_AttentionBlock(in_channels, norm_channels=n_head),
            nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1),
            VAE_ResidualBlock(inter_channels, inter_channels, norm_channels=n_head),
            VAE_ResidualBlock(inter_channels, inter_channels, norm_channels=n_head),
            VAE_AttentionBlock(inter_channels, norm_channels=n_head),
            nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1),
            VAE_ResidualBlock(inter_channels, inter_channels, norm_channels=n_head),
            VAE_ResidualBlock(inter_channels, inter_channels, norm_channels=n_head),
            VAE_AttentionBlock(inter_channels, norm_channels=n_head),
            nn.Conv2d(inter_channels, out_channels, kernel_size=3, padding=1),
            VAE_ResidualBlock(out_channels, out_channels, norm_channels=n_head),
            VAE_ResidualBlock(out_channels, out_channels, norm_channels=n_head),
            VAE_AttentionBlock(out_channels, norm_channels=n_head),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        for module in self:
            x = module(x)
        return x