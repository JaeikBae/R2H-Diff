# Standard library imports
from math import nan
import os
import datetime
import traceback
import argparse
import matplotlib.pyplot as plt
# Third-party imports
import torch
import torch.distributed
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
# Local imports
from utils.common import get_timestep_embedding, printif
from utils.dataset import HsiDataset
from ddim import get_beta_schedule
from diffusion_hsi import HsiDiffusion
# ============================================
# Argument Parsing
# ============================================
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/app/datas/hsi')
parser.add_argument('--data_dir_test', type=str, default='/app/datas/val')
parser.add_argument('--save_dir', type=str, default='../weights')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=5000)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--beta_schedule', type=str, default='cosine')
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.02)
parser.add_argument('--num_diffusion_timesteps', type=int, default=500)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--resume_path', type=str, default='')
parser.add_argument('--resume_epoch', type=int, default=None)

parser.add_argument('--core_shape', type=tuple, default=(64, 128, 128))
parser.add_argument('--note', type=str, default='no_seg')
args = parser.parse_args()

# ============================================
# Distributed Training Setup
# ============================================
dist_url = 'env://'
rank = int(os.environ['RANK'])
world_size = int(os.environ['WORLD_SIZE'])
local_rank = int(os.environ['LOCAL_RANK'])
print(f'Rank: {rank}, Local Rank: {local_rank}')
torch.cuda.set_device(local_rank)
torch.distributed.init_process_group(
    backend='nccl', init_method=dist_url,
    world_size=world_size, rank=rank
)
ISRANK_0 = (torch.distributed.get_rank() == 0)
printif(args, ISRANK_0)
torch.distributed.barrier()

# ============================================
# Constants and Device Setup
# ============================================
CORE_SHAPE = args.core_shape # (bands, height, width)
HSI_SHAPE = (512, 256, 256) # (batch_size, channels, height, width)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
printif(f'Using device: {device}', ISRANK_0)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# ============================================
# Model Architecture Setup
# ============================================
printif('Model Architecture Setup', ISRANK_0)
model = HsiDiffusion(
    HSI_SHAPE, 
    CORE_SHAPE, 
)
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=0.1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(initialize_weights)
model = model.to(device, dtype=torch.bfloat16)
model = nn.parallel.DistributedDataParallel(
    model,
    device_ids=[local_rank],
    gradient_as_bucket_view=True,
    find_unused_parameters=False
)
printif('Done', ISRANK_0)

# ============================================
# Dataset and DataLoader Setup
# ============================================
printif('Loading train dataset', ISRANK_0)
train_dataset = HsiDataset(args.data_dir, HSI_SHAPE, CORE_SHAPE, disable_tqdm=not ISRANK_0)
torch.distributed.barrier()
printif(f'Train dataset size: {len(train_dataset)}', ISRANK_0)
data_sampler = DistributedSampler(
    train_dataset, shuffle=True
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    sampler=data_sampler,
    num_workers=args.num_workers,
    persistent_workers=True
)
# ============================================
# FP16 AMP setup + DDP wrapping
# ============================================

main_params = []
f_params = []

for name, param in model.named_parameters():
    if 'f_generator' in name:
        f_params.append(param)
    else:
        main_params.append(param)

optimizer = torch.optim.Adam(main_params, lr=args.lr, weight_decay=1e-5)
optimizer_f = torch.optim.Adam(f_params, lr=args.lr, weight_decay=1e-5)

gamma = 0.9995

scheduler   = LambdaLR(optimizer,   lr_lambda=lambda epoch: gamma**epoch)
scheduler_f = LambdaLR(optimizer_f, lr_lambda=lambda epoch: gamma**epoch)
# ============================================
# Checkpoint loading
# ============================================
START_FROM, PTH_PATH, LOG_TXT = 0, '', ''
if args.resume_path.endswith('.pth'):
    START_FROM = int(args.resume_path.split('_')[-1].split('.')[0])
    PTH_PATH    = args.resume_path
    LOG_TXT     = args.resume_path.split('epoch')[0] + 'losses.txt'
elif args.resume_epoch is not None and args.resume_path:
    START_FROM = args.resume_epoch
    PTH_PATH    = os.path.join(args.resume_path, f'epoch_{START_FROM}.pth')
    LOG_TXT     = os.path.join(args.resume_path, 'losses.txt')
elif args.resume_path and os.path.exists(args.resume_path):
    START_FROM = max(
        int(f.split('_')[-1].split('.')[0])
        for f in os.listdir(args.resume_path) if f.endswith('.pth')
    )
    PTH_PATH    = os.path.join(args.save_dir, f'epoch_{START_FROM}.pth')
    LOG_TXT     = os.path.join(args.save_dir, 'losses.txt')

printif(f'Resume from epoch {START_FROM}', ISRANK_0)
printif(f'Checkpoint file: {PTH_PATH}', ISRANK_0)

losses = []
if LOG_TXT and os.path.exists(LOG_TXT):
    with open(LOG_TXT) as f:
        for line in f:
            try:
                losses.append(float(line.strip()))
            except:
                pass
losses = losses[:START_FROM]

if START_FROM > 0 and os.path.exists(PTH_PATH):
    ckpt = torch.load(PTH_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    # Load optimizer states if they exist
    if 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'optimizer_f_state_dict' in ckpt:
        optimizer_f.load_state_dict(ckpt['optimizer_f_state_dict'])
    # Load scheduler states if they exist
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if 'scheduler_f_state_dict' in ckpt:
        scheduler_f.load_state_dict(ckpt['scheduler_f_state_dict'])
    printif(f'Training from epoch {START_FROM}', ISRANK_0)
elif START_FROM > 0:
    printif(f'Checkpoint not found: {PTH_PATH}', ISRANK_0)
    START_FROM = 0
else:
    printif('Training from epoch 0', ISRANK_0)

# ============================================
# Diffusion parameters
# ============================================
T = args.num_diffusion_timesteps
betas = torch.from_numpy(
    get_beta_schedule(
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        num_diffusion_timesteps=T
    )
)

epoch_losses = {
    'main': [],
    'recon': [],
    'f': [],
}
# ============================================
# Training function
# ============================================

def safe_einsum_reconstruction(pred, f0_t, f1_t, f2_t, z0, f_0, f_1, f_2):
    f0_t_clipped = torch.clamp(f0_t, -10.0, 10.0)
    f1_t_clipped = torch.clamp(f1_t, -10.0, 10.0)
    f2_t_clipped = torch.clamp(f2_t, -10.0, 10.0)
    f_0_clipped = torch.clamp(f_0, -10.0, 10.0)
    f_1_clipped = torch.clamp(f_1, -10.0, 10.0)
    f_2_clipped = torch.clamp(f_2, -10.0, 10.0)
    
    re = torch.einsum('brhw,bcr->bchw', pred, f0_t_clipped)
    re = torch.einsum('bcrw,bhr->bchw', re, f1_t_clipped)
    re = torch.einsum('bchr,bwr->bchw', re, f2_t_clipped)
    
    re_gt = torch.einsum('brhw,bcr->bchw', z0, f_0_clipped)
    re_gt = torch.einsum('bcrw,bhr->bchw', re_gt, f_1_clipped)
    re_gt = torch.einsum('bchr,bwr->bchw', re_gt, f_2_clipped)
    
    return re, re_gt



def train(epoch, model, loader, device):
    global losses, epoch_losses
    loader.sampler.set_epoch(epoch)
    
    with tqdm(loader, desc=f'Epoch {epoch:05d}',
              total=len(loader), disable=not ISRANK_0) as pbar:
        for core, f_0, f_1, f_2, rgb, seg in loader:
            batch_losses = {
                'main': [],
                'recon': [],
                'f': [],
            }
            n = core.shape[0]
            f_0 = f_0.to(device, dtype=torch.bfloat16)
            f_1 = f_1.to(device, dtype=torch.bfloat16)
            f_2 = f_2.to(device, dtype=torch.bfloat16)
            context = torch.cat([rgb, seg], dim=1).to(device, dtype=torch.bfloat16)
            t1 = torch.randint(0, T, (n//2,))
            t2 = T - t1 - 1

            if n % 2 == 1:
                t_extra = torch.randint(0, T, (1,))
                t = torch.cat([t1, t2, t_extra], dim=0)
            else:
                t = torch.cat([t1, t2], dim=0)

            time_emb = get_timestep_embedding(t, 1280).to(device, dtype=torch.bfloat16)

            
            z0 = core
            z0 = z0.to(device, dtype=torch.bfloat16)
            
            z0 = torch.clamp(z0, -10.0, 10.0)
            f_0 = torch.clamp(f_0, -10.0, 10.0)
            f_1 = torch.clamp(f_1, -10.0, 10.0)
            f_2 = torch.clamp(f_2, -10.0, 10.0)
            a = (1 - betas).cumprod(0).index_select(0, t).view(-1,1,1,1).to(device, dtype=torch.bfloat16)
            noise = torch.randn_like(z0).to(device, dtype=torch.bfloat16)
            z_noised = z0 * torch.sqrt(a + 1e-6) + noise * torch.sqrt(1.0 - a + 1e-6)

            pred, f0_t, f1_t, f2_t = model(z_noised, context, time_emb)
            
            def stable_loss_fn(x, y):
                diff = x - y
                diff = torch.clamp(diff, -10.0, 10.0)
                return (diff ** 2).mean() * 1000
            
            pred_loss = stable_loss_fn(pred, z0)
            f0_loss = stable_loss_fn(f0_t, f_0)
            f1_loss = stable_loss_fn(f1_t, f_1)
            f2_loss = stable_loss_fn(f2_t, f_2)
            f_loss = f0_loss + f1_loss + f2_loss

            re, re_gt = safe_einsum_reconstruction(pred, f0_t, f1_t, f2_t, z0, f_0, f_1, f_2)
            recon_loss = stable_loss_fn(re, re_gt) * 1000
            
            total_loss = pred_loss + f_loss + recon_loss
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                printif(f"NaN/Inf detected in loss at epoch {epoch}, skipping batch", ISRANK_0)
                continue

            batch_losses['main'].append(pred_loss.item())
            batch_losses['recon'].append(recon_loss.item())
            batch_losses['f'].append(f_loss.item())

            optimizer.zero_grad()
            optimizer_f.zero_grad()
            
            total_loss.backward()
            
            clip_grad_norm_(main_params, max_norm=1.0)
            clip_grad_norm_(f_params, max_norm=1.0)

            optimizer.step()
            optimizer_f.step()

            main_loss = sum(batch_losses['main']) / (len(batch_losses['main']) + 1e-6)
            f_loss = sum(batch_losses['f']) / (len(batch_losses['f']) + 1e-6)
            recon_loss = sum(batch_losses['recon']) / (len(batch_losses['recon']) + 1e-6)
            lr = scheduler.get_last_lr()[0]
            

            
            pbar.set_postfix({
                'main': f'{main_loss:.6f}',
                'f': f'{f_loss:.6f}',
                'recon': f'{recon_loss:.6f}',
                'lr': f'{lr:.6f}'
            })
            pbar.update()
        total = 0
        for k, v in batch_losses.items():
            total += sum(v)
        total_loss = total / (len(batch_losses.keys()) + 1e-6)
        for k, v in batch_losses.items():
            epoch_losses[k].append(sum(v) / (len(v) + 1e-6))
        losses.append(total_loss)
    scheduler.step()
    scheduler_f.step()
    if epoch % 50 == 1 and epoch > 10:
        test(epoch, model, device)
    
    torch.cuda.empty_cache()
    torch.distributed.barrier()
# ============================================
# Test dataset and DataLoader Setup
# ============================================
torch.distributed.barrier()
printif('Loading test dataset', ISRANK_0)
test_dataset = HsiDataset(args.data_dir_test, HSI_SHAPE, CORE_SHAPE, disable_tqdm=not ISRANK_0, is_eval=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
torch.distributed.barrier()
printif(f'Test dataset size: {len(test_dataset)}', ISRANK_0)


# ============================================
# Test function
# ============================================
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def test(epoch, model, device):
    model.eval()
    T = args.num_diffusion_timesteps

    beta_t = get_beta_schedule(
        beta_schedule=args.beta_schedule, 
        beta_start=args.beta_start, 
        beta_end=args.beta_end, 
        num_diffusion_timesteps=T
        )
    n = 1
    beta_t = torch.from_numpy(beta_t)
    betav_t = torch.cat([torch.zeros(1), beta_t], dim=0) # (T+1)
    zT_t = torch.randn(n, *CORE_SHAPE).to(device, dtype=torch.bfloat16) # z0 shape: (1, 16, 32, 32)
    seq_t = range(0, T, 5)
    seq_next_t = [0] + list(seq_t[:-1])
    final_hsi_t = np.zeros((len(test_loader)*n, *HSI_SHAPE))
    for batch_idx_t, (names_t, core_t, f0_t, f1_t, f2_t, rgb_t, seg_t) in enumerate(test_loader):
        core_t = core_t.to(device, dtype=torch.bfloat16)
        f0_t = f0_t.to(device, dtype=torch.bfloat16)
        f1_t = f1_t.to(device, dtype=torch.bfloat16)
        f2_t = f2_t.to(device, dtype=torch.bfloat16)

        rgb_t = rgb_t.to(device, dtype=torch.bfloat16)
        seg_t = seg_t.to(device, dtype=torch.bfloat16)
        context_t = torch.cat([rgb_t, seg_t], dim=1)
        zt_t = zT_t.clone()
        with torch.no_grad():
            for i, j in tqdm(zip(reversed(seq_t), reversed(seq_next_t)), desc='Testing', total=len(seq_t), disable=not ISRANK_0):
                t_t = (torch.ones(1) * i)
                next_t_t = (torch.ones(1) * j)
                at_t = (1 - betav_t).cumprod(0).index_select(0, t_t.long()+1).view(-1,1,1,1)
                at_next_t = (1 - betav_t).cumprod(0).index_select(0, next_t_t.long()+1).view(-1,1,1,1)
                time_embedding_t = get_timestep_embedding(t_t, 1280)

                context_t = context_t.to(device, dtype=torch.bfloat16)
                time_embedding_t = time_embedding_t.to(device, dtype=torch.bfloat16)
                at_t = at_t.to(device, dtype=torch.bfloat16)
                at_next_t = at_next_t.to(device, dtype=torch.bfloat16)

                pred, f0_t_t, f1_t_t, f2_t_t = model(zt_t, context_t, time_embedding_t)
                
                noise_t = (zt_t - at_t.sqrt() * pred) / ((1 - at_t + 1e-6).sqrt())
                zt_t = at_next_t.sqrt() * pred + (1 - at_next_t).sqrt() * noise_t

            re = torch.einsum('brhw,bcr->bchw', pred, f0_t_t)
            re = torch.einsum('bcrw,bhr->bchw', re, f1_t_t)
            re = torch.einsum('bchr,bwr->bchw', re, f2_t_t)
            final_hsi_t[batch_idx_t*n:(batch_idx_t+1)*n] = re.detach().to(torch.float32).cpu().numpy()
            
            
    if ISRANK_0:
        re_gt = torch.einsum('brhw,bcr->bchw', core_t, f0_t)
        re_gt = torch.einsum('bcrw,bhr->bchw', re_gt, f1_t)
        re_gt = torch.einsum('bchr,bwr->bchw', re_gt, f2_t)
        
        re = final_hsi_t[batch_idx_t*n:(batch_idx_t+1)*n]
        re_gt = re_gt.detach().to(torch.float32).cpu().numpy()
        for re_i, re_gt_i, name in zip(re, re_gt, names_t):
            data_range = max(re_i.max() - re_i.min(), re_gt_i.max() - re_gt_i.min(), 1.0)
            mse_r = mean_squared_error(re_gt_i, re_i)
            psnr_r = psnr(re_gt_i, re_i, data_range=data_range)
            ssim_r = ssim(re_gt_i, re_i, data_range=data_range, multichannel=False)
            print(f"MSE: {mse_r}, PSNR: {psnr_r}, SSIM: {ssim_r}")
            with open(f'/app/test/test_hsi.txt', 'a+') as f:
                f.write(f"MSE: {mse_r}, PSNR: {psnr_r}, SSIM: {ssim_r}\n")
            
            fig, axs = plt.subplots(16, 32, figsize=(128, 64))
            fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            
            for i in range(512):
                row = i // 32
                col = i % 32
                axs[row, col].imshow(re_i[i], cmap='gray')
                axs[row, col].axis('off')
            axs[0, 0].set_title('Generated HSI (512 bands)', fontsize=12)
            
            plt.savefig(f'/app/test/test_hsi_{epoch-1}_{name}.png')
            plt.close()
    torch.distributed.barrier()
    model.train()


def save_model(epoch, time_str):
    save_path = os.path.join(args.save_dir, time_str)
    os.makedirs(save_path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_f_state_dict': optimizer_f.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_f_state_dict': scheduler_f.state_dict(),
    }, os.path.join(save_path, f'epoch_{epoch}.pth'))

    with open(os.path.join(save_path, 'losses.txt'), 'w') as f:
        for k,v in vars(args).items():
            f.write(f'{k}: {v}\n')
        f.write(f'T: {T}\n' + '-'*50 + '\n')
        for l in losses:
            f.write(f'{l}\n')
    save_file = os.path.join(save_path, 'epoch_losses.txt')
    with open(save_file, 'w') as f:
        for i in range(len(epoch_losses['main'])):
            main = epoch_losses['main'][i]
            recon = epoch_losses['recon'][i]
            f_loss = epoch_losses['f'][i]
            f.write(f"{i} {main} {recon} {f_loss}\n")
    printif(f'Model saved at epoch {epoch}', ISRANK_0)

# ============================================
# Main
# ============================================
torch.distributed.barrier()
if __name__ == '__main__':
    try:
        start_time = datetime.datetime.now().strftime('%Y%m%d%H%M')
        model.train()
        for ep in range(START_FROM+1, args.epochs+1):
            train(ep, model, train_loader, device)
            if ISRANK_0 and ep % args.save_interval == 0:
                save_model(ep, start_time)
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        if ISRANK_0:
            save_model(args.epochs, start_time)
    except Exception:
        printif(traceback.format_exc(), ISRANK_0)
