import torch
import torch.nn.functional as F
from torch import nn
from unet import HsiUnet
from encoder import HsiEncoder, RgbSegEncoder
from decoder import HsiDecoder
from fgenerator import FGenerator

class TimeEmbedding(nn.Module):
    def __init__(self, n_embd):
        super(TimeEmbedding, self).__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd) # (B, 320) -> (B, 1280)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd) # (B, 1280) -> (B, 1280)
    def forward(self, x):
        x = self.linear_1(x)
        x = F.silu(x)
        x = self.linear_2(x)
        return x



class HsiDiffusion(nn.Module):
    def __init__(
        self,
        hsi_shape,
        CORE_SHAPE,
    ):
        super().__init__()
        # ── diffusion U-Net pipeline ────────────
        M = 3
        self.encoder = HsiEncoder(CORE_SHAPE, out_channels=CORE_SHAPE[0]*M)
        self.decoder = HsiDecoder(in_channels=CORE_SHAPE[0]*M, out_channels=CORE_SHAPE[0])
        self.context_encoder = RgbSegEncoder()
        self.net             = HsiUnet(in_channels=CORE_SHAPE[0]*M)
        self.time_layer      = TimeEmbedding(1280)
        self.f_generator     = FGenerator(hsi_shape, CORE_SHAPE)

    def forward(
        self,
        z_noised: torch.Tensor,
        context: torch.Tensor,
        time_emb: torch.Tensor
    ):
        x_enc = self.encoder(z_noised)
        ctx_enc = self.context_encoder(context)
        t_emb = self.time_layer(time_emb)
        x_net = self.net(x_enc, ctx_enc, t_emb)
        x_dec = self.decoder(x_net)
        f0, f1, f2 = self.f_generator(context)
        return x_dec, f0, f1, f2