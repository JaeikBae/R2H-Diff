#!/usr/bin/env python3
"""
HSI 생성 및 분석 도구
독립적인 HSI 생성, 후처리, 성능 분석을 수행합니다.
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.cuda.amp import autocast
from tqdm import tqdm
import time
from scipy.signal import savgol_filter
import torchvision.transforms.functional as TF
from skimage.restoration import denoise_bilateral
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# 로컬 모듈 import
from diffusion_hsi import HsiDiffusion
from ddim import get_beta_schedule
from utils.common import get_timestep_embedding
from utils.dataset import HsiDataset
from utils.tucker_utils import get_optimal_core_shape
from torch.utils.data import DataLoader

class HSIGenerator:
    """HSI 생성 및 분석 클래스"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model = self._load_model(model_path)
        self.betav = get_beta_schedule('cosine', 500)
        
    def _load_model(self, model_path):
        """모델 로드"""
        print(f"모델 로딩 중: {model_path}")
        # HsiDiffusion 초기화에 필요한 인자들
        hsi_shape = (1, 512, 256, 256)  # (batch, bands, height, width)
        core_shape = (64, 128, 128)     # (bands, height, width)
        model = HsiDiffusion(hsi_shape, core_shape)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    
    def post_process_hsi(self, hsi, 
                        gaussian_kernel_size=5, 
                        gaussian_sigma=0.8,
                        bilateral_sigma_spatial=3.0,
                        bilateral_sigma_color=0.05,
                        savgol_window_length=9,
                        savgol_polyorder=2,
                        enable_gaussian=True,
                        enable_bilateral=True,
                        enable_savgol=True):
        """
        HSI 후처리 함수
        """
        hsi_processed = hsi.clone()
        
        # 1) 3D Gaussian Spatial Smoothing
        if enable_gaussian:
            hsi_batch = hsi_processed.unsqueeze(0)
            hsi_spatial = TF.gaussian_blur(hsi_batch, 
                                         kernel_size=[gaussian_kernel_size, gaussian_kernel_size], 
                                         sigma=[gaussian_sigma, gaussian_sigma])
            hsi_processed = hsi_spatial.squeeze(0)

        # 2) Bilateral Filter
        if enable_bilateral:
            hsi_processed = torch.stack([
                torch.from_numpy(denoise_bilateral(band.cpu().numpy(), 
                                                 sigma_spatial=bilateral_sigma_spatial, 
                                                 sigma_color=bilateral_sigma_color))
                for band in hsi_processed
            ]).to(self.device)

        # 3) Spectral Savitzky–Golay Filtering
        if enable_savgol:
            hsi_np = hsi_processed.cpu().numpy()
            c, h, w = hsi_np.shape
            hsi_sg = savgol_filter(hsi_np.reshape(c, -1), 
                                  window_length=savgol_window_length, 
                                  polyorder=savgol_polyorder, axis=0)
            hsi_processed = torch.from_numpy(hsi_sg.reshape(c, h, w)).to(self.device)

        return hsi_processed
    
    def generate_hsi_with_delta(self, context, delta, core_shape, hsi_shape):
        """Delta 값을 사용한 HSI 생성"""
        with torch.no_grad():
            with autocast():
                # Delta 스케줄링
                timesteps = torch.arange(0, 500, delta, device=self.device)
                num_steps = len(timesteps)
                
                # 초기 노이즈
                x = torch.randn(hsi_shape, device=self.device)
                
                # 역방향 확산 과정
                for i, t in enumerate(tqdm(timesteps.flip(0), desc=f"Delta={delta} 생성")):
                    t_tensor = t.unsqueeze(0).repeat(hsi_shape[0])
                    
                    # 예측
                    noise_pred = self.model(x, t_tensor, context)
                    
                    # 노이즈 스케줄링
                    alpha_t = 1 - self.betav[t]
                    alpha_t_prev = 1 - self.betav[t-1] if t > 0 else torch.tensor(1.0)
                    
                    # DDIM 샘플링
                    x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
                    dir_xt = torch.sqrt(1 - alpha_t_prev) * noise_pred
                    x = torch.sqrt(alpha_t_prev) * x0_pred + dir_xt
                
                return x, num_steps
    
    def calculate_metrics(self, original, generated):
        """성능 메트릭 계산"""
        orig_np = original.cpu().numpy()
        gen_np = generated.cpu().numpy()
        
        # MSE
        mse = np.mean((orig_np - gen_np) ** 2)
        
        # PSNR
        psnr = peak_signal_noise_ratio(orig_np, gen_np, data_range=orig_np.max() - orig_np.min())
        
        # SSIM (첫 번째 밴드 기준)
        ssim = structural_similarity(orig_np[0], gen_np[0], data_range=orig_np[0].max() - orig_np[0].min())
        
        return {
            'mse': mse,
            'psnr': psnr,
            'ssim': ssim
        }
    
    def analyze_delta_performance(self, context, original_hsi, delta_values, core_shape, hsi_shape):
        """Delta 값에 따른 성능 분석"""
        results = {}
        
        for delta in delta_values:
            print(f"\nDelta={delta} 분석 중...")
            
            # 생성 시간 측정
            start_time = time.time()
            generated_hsi, num_steps = self.generate_hsi_with_delta(context, delta, core_shape, hsi_shape)
            generation_time = time.time() - start_time
            
            # 후처리
            processed_hsi = self.post_process_hsi(generated_hsi)
            
            # 메트릭 계산
            raw_metrics = self.calculate_metrics(original_hsi, generated_hsi)
            processed_metrics = self.calculate_metrics(original_hsi, processed_hsi)
            
            results[delta] = {
                'generated_hsi': generated_hsi,
                'processed_hsi': processed_hsi,
                'generation_time': generation_time,
                'num_steps': num_steps,
                'mse_raw': raw_metrics['mse'],
                'psnr_raw': raw_metrics['psnr'],
                'ssim_raw': raw_metrics['ssim'],
                'mse_processed': processed_metrics['mse'],
                'psnr_processed': processed_metrics['psnr'],
                'ssim_processed': processed_metrics['ssim']
            }
            
            print(f"  생성 시간: {generation_time:.2f}초")
            print(f"  스텝 수: {num_steps}")
            print(f"  Raw MSE: {raw_metrics['mse']:.4f}, PSNR: {raw_metrics['psnr']:.2f}dB")
            print(f"  Processed MSE: {processed_metrics['mse']:.4f}, PSNR: {processed_metrics['psnr']:.2f}dB")
        
        return results
    
    def visualize_results(self, original_hsi, delta_results, save_path):
        """결과 시각화"""
        delta_values = sorted(delta_results.keys())
        mid_band = original_hsi.shape[0] // 2
        
        # 1. 이미지 비교
        fig, axes = plt.subplots(3, len(delta_values) + 1, figsize=(4*(len(delta_values) + 1), 12))
        
        # 원본
        axes[0, 0].imshow(original_hsi[mid_band].cpu(), cmap='gray')
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # 각 delta별 결과
        for i, delta in enumerate(delta_values):
            result = delta_results[delta]
            
            # Raw 결과
            axes[0, i+1].imshow(result['generated_hsi'][mid_band].cpu(), cmap='gray')
            axes[0, i+1].set_title(f'Delta={delta}\nRaw (MSE: {result["mse_raw"]:.3f})')
            axes[0, i+1].axis('off')
            
            # Processed 결과
            axes[1, i+1].imshow(result['processed_hsi'][mid_band].cpu(), cmap='gray')
            axes[1, i+1].set_title(f'Delta={delta}\nProcessed (MSE: {result["mse_processed"]:.3f})')
            axes[1, i+1].axis('off')
            
            # 차이
            diff_raw = np.abs(original_hsi[mid_band].cpu().numpy() - 
                            result['generated_hsi'][mid_band].cpu().numpy())
            axes[2, i+1].imshow(diff_raw, cmap='hot')
            axes[2, i+1].set_title(f'Delta={delta}\nDifference')
            axes[2, i+1].axis('off')
        
        # 레이블
        axes[1, 0].imshow(original_hsi[mid_band].cpu(), cmap='gray')
        axes[1, 0].set_title('Original')
        axes[1, 0].axis('off')
        
        axes[2, 0].imshow(np.zeros_like(original_hsi[mid_band].cpu()), cmap='gray')
        axes[2, 0].set_title('Original')
        axes[2, 0].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'image_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 성능 그래프
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        times = [delta_results[d]['generation_time'] for d in delta_values]
        mses_raw = [delta_results[d]['mse_raw'] for d in delta_values]
        mses_processed = [delta_results[d]['mse_processed'] for d in delta_values]
        psnrs_raw = [delta_results[d]['psnr_raw'] for d in delta_values]
        psnrs_processed = [delta_results[d]['psnr_processed'] for d in delta_values]
        steps = [delta_results[d]['num_steps'] for d in delta_values]
        
        # 생성 시간
        ax1.plot(delta_values, times, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Delta')
        ax1.set_ylabel('Generation Time (s)')
        ax1.set_title('Generation Time vs Delta')
        ax1.grid(True)
        
        # MSE 비교
        ax2.plot(delta_values, mses_raw, 'ro-', label='Raw', linewidth=2, markersize=8)
        ax2.plot(delta_values, mses_processed, 'go-', label='Processed', linewidth=2, markersize=8)
        ax2.set_xlabel('Delta')
        ax2.set_ylabel('MSE')
        ax2.set_title('MSE vs Delta')
        ax2.legend()
        ax2.grid(True)
        
        # PSNR 비교
        ax3.plot(delta_values, psnrs_raw, 'ro-', label='Raw', linewidth=2, markersize=8)
        ax3.plot(delta_values, psnrs_processed, 'go-', label='Processed', linewidth=2, markersize=8)
        ax3.set_xlabel('Delta')
        ax3.set_ylabel('PSNR (dB)')
        ax3.set_title('PSNR vs Delta')
        ax3.legend()
        ax3.grid(True)
        
        # 스텝 수
        ax4.plot(delta_values, steps, 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Delta')
        ax4.set_ylabel('Number of Steps')
        ax4.set_title('Number of Steps vs Delta')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'performance_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"시각화 결과 저장 완료: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='HSI 생성 및 분석 도구')
    parser.add_argument('--model_path', type=str, required=True, help='모델 경로')
    parser.add_argument('--data_dir', type=str, default='/app/datas/val', help='데이터 디렉토리')
    parser.add_argument('--save_dir', type=str, default='/app/results/hsi_analysis', help='결과 저장 디렉토리')
    parser.add_argument('--delta_values', type=int, nargs='+', default=[1, 2, 5, 10], help='분석할 delta 값들')
    parser.add_argument('--core_shape', type=int, nargs=3, default=[64, 128, 128], help='Core shape (bands, height, width)')
    parser.add_argument('--hsi_shape', type=int, nargs=3, default=[1, 512, 256, 256], help='HSI shape (batch, bands, height, width)')
    parser.add_argument('--device', type=str, default='cuda', help='사용할 디바이스')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드')
    
    args = parser.parse_args()
    
    # 시드 설정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 결과 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("🚀 HSI 생성 및 분석 도구 시작")
    print("=" * 60)
    print(f"📁 모델 경로: {args.model_path}")
    print(f"📁 데이터 디렉토리: {args.data_dir}")
    print(f"📁 결과 저장 디렉토리: {args.save_dir}")
    print(f"🔢 Delta 값들: {args.delta_values}")
    print(f"📊 Core shape: {args.core_shape}")
    print(f"📊 HSI shape: {args.hsi_shape}")
    print("=" * 60)
    
    # HSI 생성기 초기화
    generator = HSIGenerator(args.model_path, args.device)
    
    # 데이터 로드
    dataset = HsiDataset(args.data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 첫 번째 샘플 사용
    sample = next(iter(dataloader))
    original_hsi = sample['hsi'].squeeze(0).to(args.device)  # (C, H, W)
    context = sample['context'].to(args.device)
    
    print(f"📊 원본 HSI 크기: {original_hsi.shape}")
    print(f"📊 Context 크기: {context.shape}")
    
    # Delta 성능 분석
    results = generator.analyze_delta_performance(
        context, original_hsi, args.delta_values, 
        tuple(args.core_shape), tuple(args.hsi_shape)
    )
    
    # 결과 시각화
    generator.visualize_results(original_hsi, results, args.save_dir)
    
    # 결과 요약 저장
    summary = {
        'delta_values': args.delta_values,
        'results': {str(k): {key: float(val) if isinstance(val, (int, float)) else str(val) 
                           for key, val in v.items() if key not in ['generated_hsi', 'processed_hsi']} 
                   for k, v in results.items()}
    }
    
    import json
    with open(os.path.join(args.save_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ 분석 완료!")
    print("=" * 60)
    print("결과 해석:")
    print("- Delta가 클수록 생성 시간이 단축되지만 품질이 저하될 수 있습니다")
    print("- Delta=1이 가장 높은 품질을 제공하지만 가장 오래 걸립니다")
    print("- 후처리를 통해 품질을 개선할 수 있습니다")
    print("- 적절한 delta 값은 속도와 품질의 균형을 고려하여 선택하세요")
    print(f"📁 결과 저장 위치: {args.save_dir}")

if __name__ == "__main__":
    main() 