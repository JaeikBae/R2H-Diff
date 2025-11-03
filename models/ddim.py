# Codes below are from the official implementation of DDIM
# https://github.com/ermongroup/ddim
import torch
import numpy as np
from tqdm import tqdm
from utils.common import get_timestep_embedding, saveasimage
import matplotlib.pyplot as plt
def compute_alpha(beta, t, dtype):
    beta = torch.cat([torch.zeros(1), beta], dim=0).to(t.device, dtype=dtype)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def generalized_steps(x, seq, model, context, b, idx, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xt = x
        idx_embedding = get_timestep_embedding(idx, 160).to(x.device, dtype=x.dtype)
        for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), total=len(seq), desc="DDIM Sampling"):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long(), dtype=x.dtype)
            at_next = compute_alpha(b, next_t.long(), dtype=x.dtype)
            time_embedding = get_timestep_embedding(t, 160).to(x.device, dtype=x.dtype)
            emb = torch.cat([time_embedding, idx_embedding], dim=-1).to(x.device, dtype=x.dtype)
            et = model(xt, context, emb)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            c2 = ((1 - at_next) - 0 ** 2).sqrt()
            xt = at_next.sqrt() * x0_t + c2 * et

    return xt.to('cpu')

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        s = 0.008

        steps = np.arange(0, num_diffusion_timesteps + 1, dtype=np.float64)
        t_ratio = steps / num_diffusion_timesteps

        alphas_cumprod = np.cos(((t_ratio + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = np.zeros(num_diffusion_timesteps, dtype=np.float64)
        for i in range(num_diffusion_timesteps):
            alpha_bar_t = alphas_cumprod[i + 1]
            alpha_bar_prev = alphas_cumprod[i]
            beta_t = 1.0 - (alpha_bar_t / alpha_bar_prev)
            betas[i] = np.clip(beta_t, a_min=0.0, a_max=0.999)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas