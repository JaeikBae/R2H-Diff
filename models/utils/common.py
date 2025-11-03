import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from attention import SelfAttention

def printif(message, condition):
    if condition:
        print(message)

def saveasimage(tensor, filename):
    tensor = tensor.detach().cpu().numpy()
    # tensor = np.clip(tensor, 0, 1)
    tensor = (tensor * 255).astype(np.uint8)
    tensor = np.transpose(tensor, (1, 2, 0))
    print(tensor.shape)
    cv2.imwrite(filename, tensor)
    print(f"Image saved as {filename}")

def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models,
    but uses a different embedding function as the base.
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # add zeros
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb



def get_gpu_memory_info():
    """GPU 메모리 사용량 정보를 반환합니다."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3      # GB
        return {
            'allocated_gb': allocated,
            'cached_gb': cached,
            'free_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3 - cached
        }
    return None

def print_gpu_memory_info():
    """GPU 메모리 사용량을 출력합니다."""
    info = get_gpu_memory_info()
    if info:
        print(f"GPU Memory - Allocated: {info['allocated_gb']:.2f}GB, "
              f"Cached: {info['cached_gb']:.2f}GB, "
              f"Free: {info['free_gb']:.2f}GB")

def clear_gpu_memory():
    """GPU 메모리를 정리합니다."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels, norm_channels=4):
        super(VAE_AttentionBlock, self).__init__()
        self.attention = SelfAttention(1, channels)
        
    
    def forward(self, x):
        residue = x 
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2).contiguous()
        x = self.attention(x)
        x = F.silu(x)  # 여기에 추가
        x = x.transpose(-1, -2).contiguous()
        x = x.view((n, c, h, w))
        x += residue
        return x 

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels=4):
        super(VAE_ResidualBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x
        x = self.conv_1(x)
        x = F.silu(x)  # 다시 추가
        x = self.conv_2(x)
        x = F.silu(x)  # 다시 추가
        return x + self.residual_layer(residue)