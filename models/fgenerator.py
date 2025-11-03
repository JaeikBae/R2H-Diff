import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common import VAE_AttentionBlock, VAE_ResidualBlock

class ConvBlockEnc(nn.Module):
    def __init__(self, in_ch, out_ch, reduce=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2 if reduce else 1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class ConvBlockDec(nn.Module):
    def __init__(self, in_ch, out_ch, upsample=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Upsample(scale_factor=2) if upsample else nn.Identity(),
        )
    
    def forward(self, x):
        return self.net(x)

class FGenerator(nn.Module):
    def __init__(self, hsi_shape, core_shape):
        super().__init__()
        self.hsi_shape  = hsi_shape
        self.core_shape = core_shape
        H0, H1, H2 = self.hsi_shape
        W0, W1, W2 = self.core_shape

        M0 = 32
        M1 = 4
        M2 = 8
        M3 = 12
        def make_encoder(in_ch, mid_ch, out_ch):
            return nn.Sequential(
                ConvBlockEnc(in_ch, mid_ch),
                VAE_ResidualBlock(mid_ch, mid_ch*M1, norm_channels=4),
                VAE_AttentionBlock(mid_ch*M1, norm_channels=4),
                ConvBlockEnc(mid_ch*M1, mid_ch*M1, reduce=True),
                VAE_ResidualBlock(mid_ch*M1, mid_ch*M2, norm_channels=4),
                VAE_AttentionBlock(mid_ch*M2, norm_channels=4),
                ConvBlockEnc(mid_ch*M2, out_ch),
            )
        
        def make_decoder(in_ch, mid_ch, out_ch):
            return nn.Sequential(
                ConvBlockDec(in_ch, mid_ch*M2),
                VAE_AttentionBlock(mid_ch*M2, norm_channels=4),
                VAE_ResidualBlock(mid_ch*M2, mid_ch*M1, norm_channels=4),
                ConvBlockDec(mid_ch*M1, mid_ch*M1, upsample=True),
                VAE_AttentionBlock(mid_ch*M1, norm_channels=4),
                VAE_ResidualBlock(mid_ch*M1, mid_ch, norm_channels=4),
                ConvBlockDec(mid_ch, out_ch),
            )


        # F1, F2 (spatial) get more capacity
        self.encoder = make_encoder(1, M0, M0*M3)
        self.decoder = make_decoder(M0*M3, M0, 1)  # f0: (512, M0)

        self.h_adjust = nn.Sequential(
            ConvBlockEnc(1, 1, reduce=True),
            ConvBlockEnc(1, 1, reduce=True),
        )
        
        # Size adjustment layers for flexible core shape with gradient clipping
        self.f0_adjust = nn.Sequential(
            nn.Linear(64 * 64, H0 * W0),
            nn.Tanh()  # Bounded activation
        )
        self.f1_adjust = nn.Sequential(
            nn.Linear(64 * 64, H1 * W1),
            nn.Tanh()  # Bounded activation
        )
        self.f2_adjust = nn.Sequential(
            nn.Linear(64 * 64, H2 * W2),
            nn.Tanh()  # Bounded activation
        )
        
        # Initialize weights properly to prevent NaN
        # self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to prevent gradient explosion"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, context: torch.Tensor):
        H0, H1, H2 = self.hsi_shape
        W0, W1, W2 = self.core_shape

        # segmentation mask만 사용 (context는 rgb+seg이므로 seg 부분만 추출)
        seg = context[:, 3:, :, :]  # (B, 1, H, W)
        h = seg

        skip_connections = []
        for layer in self.encoder:
            h = layer(h)
            skip_connections.append(h)
        for layer in self.decoder:
            h = layer(h + skip_connections.pop())

        h = self.h_adjust(h)
        f0 = h.view(h.size(0), -1)
        f0 = self.f0_adjust(f0)
        f0 = f0.view(-1, H0, W0)
        f1 = h.view(h.size(0), -1)
        f1 = self.f1_adjust(f1)
        f1 = f1.view(-1, H1, W1)
        f2 = h.view(h.size(0), -1)
        f2 = self.f2_adjust(f2)
        f2 = f2.view(-1, H2, W2)


        # NaN check and replacement
        f0 = torch.nan_to_num(f0, nan=0.0, posinf=1.0, neginf=-1.0)
        f1 = torch.nan_to_num(f1, nan=0.0, posinf=1.0, neginf=-1.0)
        f2 = torch.nan_to_num(f2, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return f0, f1, f2
