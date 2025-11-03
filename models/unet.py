import torch
import torch.nn.functional as F
from torch import nn
from attention import SelfAttention, CrossAttention

class UNET_ResidualBlock(nn.Module): # handle time
    def __init__(self, in_channels, out_channels, n_time=1280*4):
        super(UNET_ResidualBlock, self).__init__()
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        residue = feature.detach()
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        time = F.silu(time)
        time = self.linear_time(time)
        
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module): # handle context
    def __init__(self, n_head: int, n_embd: int, d_context=16): # d_context from context shape (B, 64, 16, 16)
        super(UNET_AttentionBlock, self).__init__()
        channels = n_head * n_embd
        
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        residue_long = x
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2).contiguous()
        
        residue_short = x
        x = self.attention_1(x)
        x += residue_short
        
        residue_short = x
        x = self.attention_2(x, context)
        x += residue_short
        
        residue_short = x
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2).contiguous()
        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x

class HsiUnet(nn.Module):
    def __init__(self, in_channels):
        super(HsiUnet, self).__init__()
        c = in_channels
        m1 = 2
        m2 = 4
        m3 = 8
        m4 = 12
        m5 = 16

        max_heads = 16
        n_head = min(c*m1, max_heads)
        while n_head > 0 and ((c*m1 % n_head) != 0):
            n_head //= 2
        assert n_head > 0, "n_head is 0"

        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(c, c*m1, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(c*m1, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m2, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m2, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head)),
            SwitchSequential(nn.Conv2d(c*m2, c*m2, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(c*m2, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m3, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m3, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head)),
            SwitchSequential(nn.Conv2d(c*m3, c*m3, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(c*m3, c*m4), UNET_AttentionBlock(n_head, c*m4//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m4, c*m4), UNET_AttentionBlock(n_head, c*m4//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m4, c*m4), UNET_AttentionBlock(n_head, c*m4//n_head)),
            SwitchSequential(nn.Conv2d(c*m4, c*m4, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(c*m4, c*m5)),
            SwitchSequential(UNET_ResidualBlock(c*m5, c*m5)),
            SwitchSequential(UNET_ResidualBlock(c*m5, c*m5)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(c*m5, c*m5),
            UNET_AttentionBlock(n_head, c*m5//n_head),
            UNET_ResidualBlock(c*m5, c*m5),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(c*m5 + c*m5, c*m5)),
            SwitchSequential(UNET_ResidualBlock(c*m5 + c*m5, c*m5)),
            SwitchSequential(UNET_ResidualBlock(c*m5 + c*m5, c*m4)),
            SwitchSequential(UNET_ResidualBlock(c*m4 + c*m4, c*m4), UNET_AttentionBlock(n_head, c*m4//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m4 + c*m4, c*m4), UNET_AttentionBlock(n_head, c*m4//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m4 + c*m4, c*m4), UNET_AttentionBlock(n_head, c*m4//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m4 + c*m4, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m3 + c*m3, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head), Upsample(c*m3)),
            SwitchSequential(UNET_ResidualBlock(c*m3 + c*m3, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m3 + c*m3, c*m3), UNET_AttentionBlock(n_head, c*m3//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m3 + c*m3, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m2 + c*m2, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head), Upsample(c*m2)),
            SwitchSequential(UNET_ResidualBlock(c*m2 + c*m2, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m2 + c*m2, c*m2), UNET_AttentionBlock(n_head, c*m2//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m2 + c*m2, c*m1), UNET_AttentionBlock(n_head, c*m1//n_head)),
            SwitchSequential(UNET_ResidualBlock(c*m1 + c*m1, c)),
        ])

        
    def forward(self, x, context, time):
        # x : (B, 64, 16, 16)
        # context : (B, 64, 16, 16)
        # time : (B, 320*4)
        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)
        x = self.bottleneck(x, context, time)
        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1) # concat in channel
            x = layers(x, context, time)
        return x # (B, 64, 16, 16)