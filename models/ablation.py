import os
import argparse
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from diffusion_hsi import HsiDiffusion
from ddim import get_beta_schedule
from utils.common import get_timestep_embedding
from utils.dataset import HsiDataset


# =============================
# Metrics (copied/adapted from test.py)
# =============================
from skimage.metrics import structural_similarity as ssim

METRIC_EPS = 1e-8
SSIM_DATA_RANGE = 1.0
MRAE_MIN_DENOM = 1e-3
MRAE_RELERR_CLIP = 10.0


def compute_psnr(gt, pred, max_val=1.0):
    mse = np.mean((gt - pred) ** 2)
    return float(10 * np.log10((max_val ** 2) / max(mse, METRIC_EPS)))


def compute_ssim(gt, pred):
    vals = [ssim(gt[i], pred[i], data_range=SSIM_DATA_RANGE) for i in range(gt.shape[0])]
    return float(np.mean(vals))


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


    


def evaluate_metrics(gt: np.ndarray, pred: np.ndarray) -> Dict[str, Optional[float]]:
    results = {}
    try:
        results['PSNR'] = compute_psnr(gt, pred)
    except Exception:
        results['PSNR'] = None
    try:
        results['SSIM'] = compute_ssim(gt, pred)
    except Exception:
        results['SSIM'] = None
    try:
        results['RMSE'] = compute_rmse(gt, pred)
    except Exception:
        results['RMSE'] = None
    try:
        results['MRAE'] = compute_mrae(gt, pred)
    except Exception:
        results['MRAE'] = None
    try:
        results['SAM'] = compute_sam(gt, pred)
    except Exception:
        results['SAM'] = None
    return results


# =============================
# Helpers
# =============================

HSI_SHAPE = (512, 256, 256)


def parse_core_shape(s: str) -> Tuple[int, int, int]:
    s = s.strip().replace('(', '').replace(')', '')
    if 'x' in s:
        parts = s.split('x')
    else:
        parts = s.split(',')
    parts = [int(p.strip()) for p in parts]
    if len(parts) != 3:
        raise ValueError(f'invalid core shape: {s}')
    return (parts[0], parts[1], parts[2])

def _normalize_to_unit_interval(arr: np.ndarray) -> np.ndarray:
    min_val = float(arr.min())
    max_val = float(arr.max())
    return (arr - min_val) / (max_val - min_val + 1e-8)


def load_model(core_shape: Tuple[int, int, int], model_path: str, device: torch.device) -> HsiDiffusion:
    model = HsiDiffusion(HSI_SHAPE, core_shape)
    ckpt = torch.load(model_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _precompute_ground_truth(dataset: HsiDataset) -> List[Dict[str, object]]:
    cache: List[Dict[str, object]] = []
    for item_index in tqdm(range(len(dataset)), desc='Precompute GT and contexts'):
        name, core, f0, f1, f2, rgb, seg = dataset[item_index]
        # Context kept on CPU; will be moved to device per delta
        context_cpu = torch.cat([rgb.float(), seg.float()], dim=0)

        # Build GT once
        core = core.unsqueeze(0)
        f0 = f0.unsqueeze(0)
        f1 = f1.unsqueeze(0)
        f2 = f2.unsqueeze(0)
        hsi_gt = torch.einsum('brhw,bcr->bchw', core, f0)
        hsi_gt = torch.einsum('bcrw,bhr->bchw', hsi_gt, f1)
        hsi_gt = torch.einsum('bchr,bwr->bchw', hsi_gt, f2)
        hsi_gt = hsi_gt.detach().cpu().numpy()
        hsi_gt = _normalize_to_unit_interval(hsi_gt)

        cache.append({
            'name': name,
            'context': context_cpu,
            'hsi_gt': hsi_gt[0],
        })
    return cache


def compare_deltas(
    data_dir: str,
    save_base: str,
    run_name: str,
    model_path: str,
    core_shape: Tuple[int, int, int],
    deltas: List[int],
    device: torch.device,
) -> str:
    # Output dirs
    core_tag = f"{core_shape[0]}x{core_shape[1]}x{core_shape[2]}"
    run_dir = os.path.join(save_base, run_name, core_tag)
    eval_dir = os.path.join(run_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    # Dataset and model loaded once
    test_dataset = HsiDataset(data_dir, HSI_SHAPE, core_shape, is_eval=True)
    dataset_cache = _precompute_ground_truth(test_dataset)
    model = load_model(core_shape, model_path, device)
    model_dtype = next(model.parameters()).dtype

    # Time schedule (shared)
    T = 500
    beta = get_beta_schedule(
        beta_schedule='cosine', beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=T
    )
    beta_t = torch.from_numpy(beta).to(device=device, dtype=model_dtype)
    betav = torch.cat([torch.zeros(1, device=device, dtype=model_dtype), beta_t], dim=0)
    alpha_bar = (1 - betav).cumprod(0)

    metric_names = ['PSNR', 'SSIM', 'RMSE', 'MRAE', 'SAM']
    delta_results: Dict[int, Dict[str, object]] = {}

    for delta in deltas:
        seq = list(range(0, T, int(delta)))
        seq_next = [0] + seq[:-1]
        num_steps = len(seq)
        first_step = seq[0] if num_steps > 0 else None
        last_step = seq[-1] if num_steps > 0 else None
        print(f"[Delta Eval] core={core_tag} delta={delta} steps={num_steps} first={first_step} last={last_step}")

        per_sample_metrics: List[Dict[str, object]] = []

        with torch.no_grad(), autocast():
            for rec in tqdm(dataset_cache, desc=f'Delta {delta} evaluation'):
                n = 1
                context = rec['context'].unsqueeze(0).to(device=device, dtype=model_dtype)
                zt = torch.randn(n, *core_shape, device=device, dtype=model_dtype)

                # Reverse process
                for i, j in zip(reversed(seq), reversed(seq_next)):
                    t = torch.full((n,), float(i), device=device)
                    next_t = torch.full((n,), float(j), device=device)
                    at = alpha_bar.index_select(0, t.long() + 1).view(-1, 1, 1, 1)
                    at_next = alpha_bar.index_select(0, next_t.long() + 1).view(-1, 1, 1, 1)
                    time_embedding = get_timestep_embedding(t, 1280).to(device=device, dtype=model_dtype)

                    pred, f0_t, f1_t, f2_t = model(zt, context, time_embedding)
                    noise_t = (zt - at.sqrt() * pred) / ((1 - at + 1e-6).sqrt())
                    zt = at_next.sqrt() * pred + (1 - at_next).sqrt() * noise_t

                re = torch.einsum('brhw,bcr->bchw', pred, f0_t)
                re = torch.einsum('bcrw,bhr->bchw', re, f1_t)
                re = torch.einsum('bchr,bwr->bchw', re, f2_t)
                re = re.detach().cpu().numpy()
                re = _normalize_to_unit_interval(re)

                gt_np = rec['hsi_gt']
                np.save(os.path.join(eval_dir, f'{rec["name"]}_{delta}_gt.npy'), gt_np)
                np.save(os.path.join(eval_dir, f'{rec["name"]}_{delta}_re.npy'), re[0])
                metrics = evaluate_metrics(gt_np, re[0])
                per_sample_metrics.append({
                    'sample_name': rec['name'],
                    'metrics': metrics,
                })

        # Aggregate
        avg_metrics: Dict[str, Optional[float]] = {}
        for m in metric_names:
            vals = [r['metrics'].get(m) for r in per_sample_metrics if r['metrics'].get(m) is not None]
            avg_metrics[m] = float(np.mean(vals)) if len(vals) > 0 else None

        delta_results[int(delta)] = {
            'Average': avg_metrics,
            'PerSample': per_sample_metrics,
            'num_steps': num_steps,
            'first_step': first_step,
            'last_step': last_step,
            'num_items_total': len(dataset_cache),
        }

    summary = {
        'ByDelta': delta_results,
        'Config': {
            'data_dir': data_dir,
            'model': model_path,
            'run_name': run_name,
            'core_shape': core_shape,
            'deltas': deltas,
        }
    }

    out_path = os.path.join(eval_dir, 'summary_by_delta.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/app/datas/hsi')
    parser.add_argument('--save_base', type=str, default='/app/results')
    parser.add_argument('--run_name', type=str, default='ablation')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--core_shape', type=str, required=True, help='Core shape, e.g., 64x128x128')
    parser.add_argument('--deltas', type=str, default='5,10,25,50')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    deltas = [int(x.strip()) for x in args.deltas.split(',') if x.strip()]
    core_shape = parse_core_shape(args.core_shape)
    compare_deltas(
        data_dir=args.data_dir,
        save_base=args.save_base,
        run_name=args.run_name,
        model_path=args.model,
        core_shape=core_shape,
        deltas=deltas,
        device=device,
    )


if __name__ == '__main__':
    main()


"""
python3 ablation.py \
  --data_dir /app/datas/hsi \
  --save_base /app/results \
  --run_name delta_only_20250814 \
  --model /app/weights/202508070233/epoch_2000.pth \
  --core_shape 64x128x128 \
  --deltas 1,5,10,25,50,100,250,500
"""