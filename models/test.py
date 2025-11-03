# %%
import os
import time
import argparse
import torch
from PIL import Image
import numpy as np
from diffusion_hsi import HsiDiffusion
from torch.cuda.amp import autocast
from ddim import get_beta_schedule
from utils.common import get_timestep_embedding
from utils.dataset import HsiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from upsample_model import UpsampleModel
from skimage.metrics import structural_similarity as ssim

METRIC_EPS = 1e-8
SSIM_DATA_RANGE = 1.0
MRAE_MIN_DENOM = 1e-3
MRAE_RELERR_CLIP = 10.0


# %%
print('=' * 60)
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/app/datas/hsi')
parser.add_argument('--model', type=str, default='/app/weights/diffusion_model.pth')
parser.add_argument('--upsample_model', type=str, default='/app/weights/upsample_model.pth')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--run_name', type=str, default=f'{time.strftime("%Y%m%d%H%M")}')
parser.add_argument('--save_base', type=str, default='/app/results')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.02)
parser.add_argument('--beta_schedule', type=str, default='cosine')
parser.add_argument('--num_diffusion_timesteps', type=int, default=500)
parser.add_argument('--core_shape', type=tuple, default=(64, 128, 128))
try:
    args = parser.parse_args()
except SystemExit: 
    # for jupyter notebook execution
    # not error even if error printed in jupyter notebook
    args = parser.parse_args(args=[])
for arg in args.__dict__:
    print(arg, getattr(args, arg))
print('=' * 60)

if not os.path.exists(args.data_dir):
    raise FileNotFoundError(f'data path not found: {args.data_dir}')

CORE_SHAPE = args.core_shape # (bands, height, width)
HSI_SHAPE = (512, 256, 256) # (batch_size, channels, height, width)
print('CORE_SHAPE: ', CORE_SHAPE)
print('HSI_SHAPE: ', HSI_SHAPE)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
        
print('=' * 60)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print('=' * 60)
# %%
print('=' * 60)
print('Loading model...')
print('=' * 60)
model = HsiDiffusion(HSI_SHAPE, CORE_SHAPE)
checkpoint = torch.load(args.model, map_location='cpu')
if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
else:
    model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print('=' * 60)
print('Model loaded successfully!')
print('=' * 60)
# %%
print('=' * 60)
print('Loading Upsample model...')
print('=' * 60)

upsample_model = UpsampleModel()
state_dict = torch.load(args.upsample_model, map_location='cpu')['model_state_dict']

upsample_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})

upsample_model = upsample_model.float()
upsample_model.to(device)
upsample_model.eval()

print(f"Upsample model loaded: {args.upsample_model}")
print('=' * 60)
# %%
print('Target dataset: ', args.data_dir)
for data in os.listdir(args.data_dir):
    print("    └", data)

test_dataset = HsiDataset(args.data_dir, HSI_SHAPE, CORE_SHAPE, is_eval=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('=' * 60)
print('Dataset loaded successfully!')
print('=' * 60)

# %%
T = args.num_diffusion_timesteps
beta = get_beta_schedule(
    beta_schedule=args.beta_schedule, 
    beta_start=args.beta_start, 
    beta_end=args.beta_end, 
    num_diffusion_timesteps=T
    )
beta = torch.from_numpy(beta)
betav = torch.cat([torch.zeros(1), beta], dim=0) # (T+1)
delta = 25
seq = range(0, T, delta)
seq_next = [0] + list(seq[:-1])

print('=' * 60)
print("Betav: ", betav.shape)
print("Delta: ", delta)
print('=' * 60)
# %%
print('=' * 60)
print('Testing...')
print('=' * 60)

RUN_DIR = os.path.join(args.save_base, args.run_name)
EXPORT_DIR = os.path.join(RUN_DIR, 'exports')
EVAL_SAVE_DIR = os.path.join(RUN_DIR, 'eval')
for d in [RUN_DIR, EXPORT_DIR, EVAL_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# %%
all_names = []
all_results = []
for batch_idx, (names, core, f_0, f_1, f_2, rgb, seg) in tqdm(enumerate(test_loader), desc='Testing', position=0, total=len(test_loader)):

    rgb = rgb.to(device).float()
    seg = seg.to(device).float()
    context = torch.cat([rgb, seg], dim=1)
    n = 1
    zt = torch.randn(n, *CORE_SHAPE).to(device).float()
    with autocast():
        with torch.no_grad():
            for i, j in tqdm(zip(reversed(seq), reversed(seq_next)), desc='Sampling', position=1, leave=False, total=len(seq)):
                t = (torch.ones(n) * i)
                next_t = (torch.ones(n) * j)
                at = (1 - betav).cumprod(0).index_select(0, t.long()+1).view(-1,1,1,1)
                at_next = (1 - betav).cumprod(0).index_select(0, next_t.long()+1).view(-1,1,1,1)
                time_embedding = get_timestep_embedding(t, 1280)

                context = context.to(device).float()
                time_embedding = time_embedding.to(device).float()
                at = at.to(device).float()
                at_next = at_next.to(device).float()

                pred, f0_t, f1_t, f2_t = model(zt, context, time_embedding)
                noise_t = (zt - at.sqrt() * pred) / ((1 - at + 1e-6).sqrt())
                zt = at_next.sqrt() * pred + (1 - at_next).sqrt() * noise_t
            re = torch.einsum('brhw,bcr->bchw', pred, f0_t)
            re = torch.einsum('bcrw,bhr->bchw', re, f1_t)
            re = torch.einsum('bchr,bwr->bchw', re, f2_t)
            re = re.detach().cpu().numpy()
            re = (re - re.min()) / (re.max() - re.min())

            with autocast():
                re_tensor = torch.from_numpy(re).float().to(device)
                re_processed = upsample_model(re_tensor)
                re_processed = re_processed.detach().cpu().numpy()
                re_processed = (re_processed - re_processed.min()) / (re_processed.max() - re_processed.min())
            
            for i in range(len(re)):
                current_data_dir = os.path.join(args.data_dir, names[i] if isinstance(names, (list, tuple)) else names)
                hsi_files = sorted(
                    [f for f in os.listdir(current_data_dir) if f.endswith('.bmp')],
                    key=lambda x: int(x.split('nm')[0].split('_')[-1])
                )
                hsi_stack = np.stack([
                    np.array(
                        Image.open(os.path.join(current_data_dir, fn)).convert('L')
                            .resize((HSI_SHAPE[2], HSI_SHAPE[1]))
                    ) for fn in hsi_files
                ], axis=0)  # uint8
                
                hsi_gt = (hsi_stack - hsi_stack.min()) / (hsi_stack.max() - hsi_stack.min())
                
                
                sample_name = names[i] if isinstance(names, (list, tuple)) else names

                all_results.append({
                    'gt_matched': re_processed[i],
                    'hsi_gt': hsi_gt,
                    'sample_name': sample_name
                })
            if isinstance(names, str):
                names = [names]
            all_names.extend(list(names))
# %%
print('=' * 60)
print('Test completed successfully!')
print('=' * 60)
print(f'Total samples processed: {len(all_names)}')
print(f'Results saved to: {EVAL_SAVE_DIR}')
print('=' * 60)
os.makedirs(EVAL_SAVE_DIR, exist_ok=True)
np.save(os.path.join(EVAL_SAVE_DIR, 'all_results.npy'), all_results)
# %%
print('=' * 60)
print('Performance Evaluation...')
print('=' * 60)


def compute_psnr(gt, pred, max_val=1.0):
    mse = np.mean((gt - pred) ** 2)
    return float(10 * np.log10((max_val ** 2) / max(mse, METRIC_EPS)))

def compute_ssim(gt, pred):
    s = [ssim(gt[i], pred[i], data_range=SSIM_DATA_RANGE) for i in range(gt.shape[0])]
    return float(np.mean(s))

def compute_rmse(gt, pred):
    return float(np.sqrt(np.mean((gt - pred) ** 2)))

def compute_mrae(gt, pred, eps=MRAE_MIN_DENOM):
    denom = np.maximum(np.abs(gt), eps)
    rel_err = np.abs(gt - pred) / denom
    if MRAE_RELERR_CLIP is not None:
        rel_err = np.clip(rel_err, 0.0, MRAE_RELERR_CLIP)
    return float(np.mean(rel_err))

def compute_sam(gt, pred, eps=1e-8):
    gt_flat = gt.reshape(gt.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    dot = np.sum(gt_flat * pred_flat, axis=0)
    norm = np.linalg.norm(gt_flat, axis=0) * np.linalg.norm(pred_flat, axis=0) + eps
    angle = np.arccos(np.clip(dot / norm, -1, 1))
    return float(np.mean(angle))

def compute_precision_recall(gt, pred, threshold=0.1):
    error = np.abs(gt - pred)
    correct_pixels = error < threshold
    total_pixels = gt.size
    precision = np.sum(correct_pixels) / total_pixels if total_pixels > 0 else 0
    recall = precision
    return float(precision), float(recall)

def evaluate_hsi_comprehensive(gt, pred):
    results = {}
    results['PSNR'] = compute_psnr(gt, pred)
    results['SSIM'] = compute_ssim(gt, pred)
    results['RMSE'] = compute_rmse(gt, pred)
    results['MRAE'] = compute_mrae(gt, pred)
    results['SAM'] = compute_sam(gt, pred)
    return results


total_gt_matched_mse = 0
performance_results = {}

for i in range(len(all_names)):
    print(f'Evaluating {i+1}/{len(all_names)}: {all_names[i]}')

    final_hsi_gt_matched = torch.from_numpy(all_results[i]['gt_matched']).float()
    hsi_tensor = torch.from_numpy(all_results[i]['hsi_gt']).float()
    
    gt_matched_mse = ((final_hsi_gt_matched - hsi_tensor) ** 2).mean()
    
    total_gt_matched_mse += gt_matched_mse
    
    gt_np = hsi_tensor.numpy()
    gt_matched_np = final_hsi_gt_matched.numpy()

    gt_matched_metrics = evaluate_hsi_comprehensive(gt_np, gt_matched_np)
    
    performance_results[all_names[i]] = {
        "GT_Matched_MSE": float(gt_matched_mse),
        "GT_Matched_Metrics": gt_matched_metrics
    }   

avg_gt_matched_mse = total_gt_matched_mse / len(all_names)

avg_gt_matched_metrics = {}

metric_names = ['PSNR', 'SSIM', 'RMSE', 'MRAE', 'SAM']

for metric in metric_names:
    gt_matched_values = []
    
    for sample_name in all_names:
        if metric in performance_results[sample_name]['GT_Matched_Metrics'] and performance_results[sample_name]['GT_Matched_Metrics'][metric] is not None:
            gt_matched_values.append(performance_results[sample_name]['GT_Matched_Metrics'][metric])
    
    avg_gt_matched_metrics[metric] = float(np.mean(gt_matched_values))
# %%
performance_results["Average_GT_Matched_MSE"] = float(avg_gt_matched_mse)
performance_results["Average_GT_Matched_Metrics"] = avg_gt_matched_metrics
performance_results["Config"] = {
    "data_dir": args.data_dir,
    "model": args.model,
    "run_name": args.run_name,
    "save_base": args.save_base,
        "upsample_model": args.upsample_model,
}
# %%
import json

os.makedirs(args.save_base, exist_ok=True)
EVAL_RESULT_FILE = os.path.join(args.save_base, f'{args.run_name}.json')
EVAL_RESULT_FILE_IN_RUN = os.path.join(EVAL_SAVE_DIR, 'summary.json')

try:
    with open(EVAL_RESULT_FILE_IN_RUN, 'w') as f:
        json.dump(performance_results, f, indent=2)
    print(f"Evaluation results saved to {EVAL_RESULT_FILE} and {EVAL_RESULT_FILE_IN_RUN}")
except Exception as e:
    print(f"Evaluation results save failed: {str(e)}")

print('=' * 60)
print('Overall Performance Summary')
print('=' * 60)
print(f'Average GT Matched MSE: {avg_gt_matched_mse:.6f}')

print('\n=== Comprehensive Evaluation Metrics (GT Matched vs GT) ===')
for metric, value in avg_gt_matched_metrics.items():
    print(f'Average GT Matched {metric}: {value:.6f}')
# %%
