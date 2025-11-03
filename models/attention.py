import torch
from torch import nn
from torch.nn import functional as F
import math
from flash_attn import flash_attn_func

class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super(SelfAttention, self).__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, causal_mask=False):
        input_shape = x.shape 
        batch_size, sequence_length, _ = input_shape 

        # Project input to query, key, value
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # Reshape for flash attention
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_head)
        k = k.view(batch_size, sequence_length, self.n_heads, self.d_head)
        v = v.view(batch_size, sequence_length, self.n_heads, self.d_head)

        # Use flash attention
        output = flash_attn_func(q, k, v, causal=causal_mask)
        
        # Reshape back to original shape
        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output

class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super(CrossAttention, self).__init__()
        self.q_proj   = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj   = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        batch_size, sequence_length, _ = x.shape
        
        # Project inputs
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        # Reshape for flash attention
        q = q.view(batch_size, sequence_length, self.n_heads, self.d_head)
        k = k.view(batch_size, -1, self.n_heads, self.d_head)
        v = v.view(batch_size, -1, self.n_heads, self.d_head)
        
        # Use flash attention
        output = flash_attn_func(q, k, v, causal=False)
        
        # Reshape back to original shape
        output = output.reshape(batch_size, sequence_length, -1)
        output = self.out_proj(output)
        
        return output