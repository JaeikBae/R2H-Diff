# %%
import os
import argparse
import torch
from PIL import Image
import numpy as np
from diffusion_hsi import HsiDiffusion
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from ddim import get_beta_schedule
from utils.common import get_timestep_embedding
from utils.dataset import HsiDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from large_gt_matching_model import LargeGTMatchingModel
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.linalg import sqrtm

# 상수 정의
DEFAULT_GT_MATCHING_MODEL_PATH = '/app/weights/align_202508071057/large_gt_matching_best.pth'  # 최고 성능 모델 사용
DEFAULT_SAVE_BASE = '/app/results'
DEFAULT_EVAL_BASE = '/app/evaluation/result_jsons'

COMPARE_CHANNELS = [0, 127, 255, 383, 511]  # 비교할 채널 인덱스
NUM_ROWS = 3  # Original, Generated, GT Matched
ROW_TITLES = ['Original', 'Generated', 'GT Matched']

# 평가 지표 관련 상수 (수치적 안정성과 일관성 보장)
METRIC_EPS = 1e-8
SSIM_DATA_RANGE = 1.0  # 입력이 0~1 정규화되어 있으므로 고정 범위 사용
MRAE_MIN_DENOM = 1e-3  # gt가 0 근처일 때 분모 안정화
MRAE_RELERR_CLIP = 10.0  # 상대오차 상한(이상치 완화)
FID_EPS = 1e-6  # 공분산 안정화 항

# 평가 지표 함수들
def compute_psnr(gt, pred, max_val=1.0):
    """PSNR (Peak Signal-to-Noise Ratio) 계산"""
    mse = np.mean((gt - pred) ** 2)
    # 0 division 방지 및 수치 안정화
    return float(10 * np.log10((max_val ** 2) / max(mse, METRIC_EPS)))

def compute_ssim(gt, pred):
    """SSIM (Structural Similarity Index) 계산"""
    # 각 채널을 2D 이미지로 간주하여 고정 data_range로 일관성 있는 SSIM을 계산
    s = [ssim(gt[i], pred[i], data_range=SSIM_DATA_RANGE) for i in range(gt.shape[0])]
    return float(np.mean(s))

def compute_rmse(gt, pred):
    """RMSE (Root Mean Square Error) 계산"""
    return float(np.sqrt(np.mean((gt - pred) ** 2)))

def compute_mrae(gt, pred, eps=MRAE_MIN_DENOM):
    """MRAE (Mean Relative Absolute Error) 계산"""
    # 분모를 충분히 크게 만들어 0 근처에서의 폭주 방지
    denom = np.maximum(np.abs(gt), eps)
    rel_err = np.abs(gt - pred) / denom
    # 이상치 완화(평균 왜곡 방지)
    if MRAE_RELERR_CLIP is not None:
        rel_err = np.clip(rel_err, 0.0, MRAE_RELERR_CLIP)
    return float(np.mean(rel_err))

def compute_sam(gt, pred, eps=1e-8):
    """SAM (Spectral Angle Mapper) 계산 - HSI 전용"""
    gt_flat = gt.reshape(gt.shape[0], -1)
    pred_flat = pred.reshape(pred.shape[0], -1)
    dot = np.sum(gt_flat * pred_flat, axis=0)
    norm = np.linalg.norm(gt_flat, axis=0) * np.linalg.norm(pred_flat, axis=0) + eps
    angle = np.arccos(np.clip(dot / norm, -1, 1))
    return float(np.mean(angle))

def compute_fid(gt, pred, eps=FID_EPS):
    """
    FID (Fréchet Inception Distance) 계산
    HSI 데이터에 맞게 수정된 버전
    """
    try:
        # 데이터를 2D로 평탄화 (채널별로)
        gt_flat = gt.reshape(gt.shape[0], -1)  # (C, H*W)
        pred_flat = pred.reshape(pred.shape[0], -1)  # (C, H*W)
        
        # 평균과 공분산 계산
        mu_gt = np.mean(gt_flat, axis=1)
        mu_pred = np.mean(pred_flat, axis=1)
        
        sigma_gt = np.cov(gt_flat)
        sigma_pred = np.cov(pred_flat)
        
        # 대각선에 작은 값 추가하여 안정성 확보
        sigma_gt += np.eye(sigma_gt.shape[0]) * eps
        sigma_pred += np.eye(sigma_pred.shape[0]) * eps
        
        # 평균 차이의 제곱
        diff = mu_gt - mu_pred
        mean_term = np.sum(diff ** 2)
        
        # 공분산 행렬의 제곱근 계산
        covmean = sqrtm(sigma_gt @ sigma_pred)
        
        # 허수부가 있으면 실수부만 사용
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # 수치 불안정으로 NaN/Inf 발생 시 안전값으로 대체
        covmean = np.nan_to_num(covmean, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 공분산 항 계산
        cov_term = np.trace(sigma_gt + sigma_pred - 2 * covmean)
        
        fid = mean_term + cov_term
        # 음수로 내려가는 수치 오차 보정
        return float(max(fid, 0.0))
    except Exception as e:
        print(f"FID 계산 중 오류 발생: {e}")
        return None

def compute_precision_recall(gt, pred, threshold=0.1):
    """
    Precision과 Recall 계산
    HSI 데이터에 맞게 수정된 버전
    """
    try:
        # 오차 계산
        error = np.abs(gt - pred)
        
        # 임계값을 기준으로 정확/부정확 판단
        correct_pixels = error < threshold
        incorrect_pixels = error >= threshold
        
        # 전체 픽셀 수
        total_pixels = gt.size
        
        # Precision: 정확한 픽셀 중 실제로 정확한 비율
        precision = np.sum(correct_pixels) / total_pixels if total_pixels > 0 else 0
        
        # Recall: 실제 정확한 픽셀 중 정확하게 예측된 비율
        # 여기서는 precision과 동일하게 계산 (전체 픽셀 대비 정확한 픽셀 비율)
        recall = precision
        
        return float(precision), float(recall)
    except Exception as e:
        print(f"Precision/Recall 계산 중 오류 발생: {e}")
        return None, None

def evaluate_hsi_comprehensive(gt, pred):
    """포괄적인 HSI 평가 지표 계산"""
    results = {}
    
    try:
        results['PSNR'] = compute_psnr(gt, pred)
    except:
        results['PSNR'] = None
    
    try:
        results['SSIM'] = compute_ssim(gt, pred)
    except:
        results['SSIM'] = None
    
    try:
        results['RMSE'] = compute_rmse(gt, pred)
    except:
        results['RMSE'] = None
    
    try:
        results['MRAE'] = compute_mrae(gt, pred)
    except:
        results['MRAE'] = None
    
    try:
        results['SAM'] = compute_sam(gt, pred)
    except:
        results['SAM'] = None
    
    try:
        results['FID'] = compute_fid(gt, pred)
    except:
        results['FID'] = None
    
    return results


class VisualizationHelper:
    """테스트 결과 시각화를 위한 헬퍼 클래스"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_comparison(self, input_data, output_data, gt_data, sample_name):
        """
        입력/출력/원본을 비교하여 시각화 (5개 채널, 5x3 형태)
        
        Args:
            input_data: 모델 입력 데이터 (B, C, H, W)
            output_data: 모델 출력 데이터 (B, C, H, W)
            gt_data: Ground Truth 데이터 (B, C, H, W)
            sample_name: 샘플 이름
            batch_idx: 배치 인덱스
            sample_idx: 샘플 인덱스
        """
        
        # 채널 선택: 전역 상수 사용
        channels = COMPARE_CHANNELS
        
        # len(channels) x 3 형태로 시각화
        fig, axes = plt.subplots(len(channels), 3, figsize=(12, 4 * len(channels)))
        
        for i, ch in enumerate(channels):
            # 입력 데이터
            axes[i, 0].imshow(input_data[ch], cmap='gray', vmin=0, vmax=1)
            axes[i, 0].set_title(f'Input Ch{ch} ({sample_name})')
            axes[i, 0].axis('off')
            
            # 출력 데이터
            axes[i, 1].imshow(output_data[ch], cmap='gray', vmin=0, vmax=1)
            axes[i, 1].set_title(f'Output Ch{ch} ({sample_name})')
            axes[i, 1].axis('off')
            
            # Ground Truth
            axes[i, 2].imshow(gt_data[ch], cmap='gray', vmin=0, vmax=1)
            axes[i, 2].set_title(f'GT Ch{ch} ({sample_name})')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # 파일 저장
        filename = f'comparison_{sample_name}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
    def visualize_spectrum_comparison(self, input_data, output_data, gt_data, sample_name):
        """
        스펙트럼 비교 시각화 (특정 픽셀의 스펙트럼)
        """
        # CPU로 이동하고 numpy로 변환
        # 중앙 픽셀의 스펙트럼 사용
        center_h, center_w = input_data.shape[1] // 2, input_data.shape[2] // 2
        input_np = input_data[:, center_h, center_w]
        output_np = output_data[:, center_h, center_w]
        gt_np = gt_data[:, center_h, center_w]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # 스펙트럼 비교
        wavelengths = np.arange(512)  # 512개 채널
        
        axes[0].plot(wavelengths, input_np, 'b-', alpha=0.7, label='Input')
        axes[0].plot(wavelengths, output_np, 'r-', alpha=0.7, label='Output')
        axes[0].plot(wavelengths, gt_np, 'g-', alpha=0.7, label='Ground Truth')
        axes[0].set_xlabel('Channel')
        axes[0].set_ylabel('Intensity')
        axes[0].set_title(f'Spectrum Comparison ({sample_name})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 차이 분석
        input_diff = np.abs(input_np - gt_np)
        output_diff = np.abs(output_np - gt_np)
        
        axes[1].plot(wavelengths, input_diff, 'b-', alpha=0.7, label='Input Error')
        axes[1].plot(wavelengths, output_diff, 'r-', alpha=0.7, label='Output Error')
        axes[1].set_xlabel('Channel')
        axes[1].set_ylabel('Absolute Error')
        axes[1].set_title(f'Error Analysis ({sample_name})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 파일 저장
        filename = f'spectrum_{sample_name}.png'
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"스펙트럼 시각화 저장: {filename}")
            

# %%
print('=' * 60)
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/app/datas/hsi')
parser.add_argument('--model', type=str, default='/app/weights/202508070233/epoch_2000.pth')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--run_name', type=str, default='align_202508071057_202508070233_epoch_2000_test')
parser.add_argument('--save_base', type=str, default=DEFAULT_SAVE_BASE)
parser.add_argument('--eval_base', type=str, default=DEFAULT_EVAL_BASE)
parser.add_argument('--gt_matching', type=str, default=DEFAULT_GT_MATCHING_MODEL_PATH, help='GT matching model path or "none" to disable')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.02)
parser.add_argument('--beta_schedule', type=str, default='cosine')
parser.add_argument('--num_diffusion_timesteps', type=int, default=500)
parser.add_argument('--core_shape', type=tuple, default=(64, 128, 128))
# 일반 CLI에서는 인자를 정상 파싱, 주피터 등에서 충돌 시 빈 인자 사용
try:
    args = parser.parse_args()
except SystemExit:
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
        
# GPU 개수 확인
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
# GT Matching 모델 로드
print('=' * 60)
print('Loading GT Matching model...')
print('=' * 60)

def load_gt_matching_model(model_path, device):
    """GT Matching 모델 로드 함수"""
    try:
        model = LargeGTMatchingModel()
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint['model_state_dict']
        
        # DataParallel 모델 고려하여 state_dict 정리
        if any(key.startswith('module.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # 'module.' 제거
                else:
                    new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        
        # 모델을 float32로 변환하여 데이터 타입 일치
        model = model.float()
        model.to(device)
        model.eval()
        
        print(f"GT Matching 모델 로드 성공: {model_path}")
        print(f"모델 데이터 타입: {next(model.parameters()).dtype}")
        return model
        
    except Exception as e:
        print(f"GT Matching 모델 로드 실패: {e}")
        return None

gt_matching_model = None
if args.gt_matching and str(args.gt_matching).lower() != 'none':
    gt_matching_model = load_gt_matching_model(args.gt_matching, device)
    if gt_matching_model is None:
        print("GT Matching 모델을 사용할 수 없습니다. 기본 생성 결과만 사용합니다.")
else:
    print('GT Matching 비활성화됨 (--gt_matching none)')
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
delta = 50
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

# 결과 저장 루트 구성
RUN_DIR = os.path.join(args.save_base, args.run_name)
VIS_DIR = os.path.join(RUN_DIR, 'visualizations')
EXPORT_DIR = os.path.join(RUN_DIR, 'exports')
EVAL_SAVE_DIR = os.path.join(RUN_DIR, 'eval')
for d in [RUN_DIR, VIS_DIR, EXPORT_DIR, EVAL_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

# 시각화 헬퍼 초기화
visualizer = VisualizationHelper(VIS_DIR)
# %%
all_names = []
res = []
res_gt_matched = []
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
            # GT Matching 모델 학습 시와 동일한 정규화 방식 적용
            re = (re - re.min()) / (re.max() - re.min())
            # ===================================================================
            try:
                POST_TRAIN_SAVE_DIR = '/app/align_data'
                os.makedirs(POST_TRAIN_SAVE_DIR, exist_ok=True)
                save_name = str(names[i]) if isinstance(names, (list, tuple)) else str(names)
                npy_save_path = os.path.join(POST_TRAIN_SAVE_DIR, f'{save_name}_{delta}.npy')
                np.save(npy_save_path, re[i])
                # ori를 float16으로 변환하여 저장
                ori = torch.einsum('brhw,bcr->bchw', core, f_0)
                ori = torch.einsum('bcrw,bhr->bchw', ori, f_1)
                ori = torch.einsum('bchr,bwr->bchw', ori, f_2)
                ori = ori.detach().cpu().numpy()
                ori = (ori - ori.min()) / (ori.max() - ori.min())
                # float16으로 변환하여 저장
                ori_float16 = ori.astype(np.float16)
                npy_save_path = os.path.join(POST_TRAIN_SAVE_DIR, f'{save_name}_{delta}_gt.npy')
                np.save(npy_save_path, ori_float16[i])
                continue
            except Exception as e:
                print(f"[Error] Failed to save npy for {save_name}: {e}")
                continue
            # ===================================================================
            # GT Matching 모델 적용 (옵션)
            if gt_matching_model is not None:
                with autocast():
                    re_tensor = torch.from_numpy(re).float().to(device)
                    re_processed = gt_matching_model(re_tensor)
                    re_processed = re_processed.detach().cpu().numpy()
                    re_processed = (re_processed - re_processed.min()) / (re_processed.max() - re_processed.min())
            else:
                re_processed = re.copy()
            
            for i in range(len(re)):
                res.append(re[i])
                res_gt_matched.append(re_processed[i])
                
                # 시각화 적용 (첫 번째 샘플만)
                # 원본 데이터 로드
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
                
                # GT Matching 모델 학습 시와 동일한 정규화 방식 적용
                hsi_gt = (hsi_stack - hsi_stack.min()) / (hsi_stack.max() - hsi_stack.min())
                
                # 생성 결과와 GT Matching 결과를 텐서로 변환 (이미 0-1 범위)
                
                # 시각화 실행
                sample_name = names[i] if isinstance(names, (list, tuple)) else names
                # visualizer.visualize_comparison(
                #     re[i],
                #     re_processed[i],
                #     hsi_gt,
                #     sample_name
                # )
                # visualizer.visualize_spectrum_comparison(
                #     re[i],
                #     re_processed[i],
                #     hsi_gt,
                #     sample_name
                # )
            if isinstance(names, str):
                names = [names]
            all_names.extend(list(names))
# %%
print('=' * 60)
print('Test completed successfully!')
print('=' * 60)
print(f'Total samples processed: {len(all_names)}')
print(f'Results saved to: {VIS_DIR}')
print('=' * 60)

# %%
print('=' * 60)
print('Performance Evaluation...')
print('=' * 60)

# 전체 성능 평가
total_original_mse = 0
total_gt_matched_mse = 0

# 전체 성능 평가 결과 저장 딕셔너리
performance_results = {}

for i in range(len(all_names)):
    print(f'Evaluating {i+1}/{len(all_names)}: {all_names[i]}')
    
    # 원본 데이터 로드
    current_data_dir = os.path.join(args.data_dir, all_names[i])
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
    hsi_tensor = torch.from_numpy(hsi_stack).float()
    
    # GT Matching 모델 학습 시와 동일한 정규화 방식 적용
    hsi_tensor = (hsi_tensor - hsi_tensor.min()) / (hsi_tensor.max() - hsi_tensor.min())
    
    # 생성 결과와 GT Matching 결과 (이미 0-1 범위)
    final_hsi_tensor = torch.from_numpy(res[i]).float()
    final_hsi_gt_matched = torch.from_numpy(res_gt_matched[i]).float()
    
    # 성능 평가 (기존 MSE)
    original_mse = ((final_hsi_tensor - hsi_tensor) ** 2).mean()
    gt_matched_mse = ((final_hsi_gt_matched - hsi_tensor) ** 2).mean()
    
    total_original_mse += original_mse
    total_gt_matched_mse += gt_matched_mse
    
    # 포괄적인 평가 지표 계산
    gt_np = hsi_tensor.numpy()
    original_np = final_hsi_tensor.numpy()
    gt_matched_np = final_hsi_gt_matched.numpy()

    # Original vs GT 평가
    original_metrics = evaluate_hsi_comprehensive(gt_np, original_np)
    
    # GT Matched vs GT 평가
    gt_matched_metrics = evaluate_hsi_comprehensive(gt_np, gt_matched_np)
    
    # 개별 샘플 결과 저장
    performance_results[all_names[i]] = {
        "Original_MSE": float(original_mse),
        "GT_Matched_MSE": float(gt_matched_mse),
        "Improvement(%)": float(((original_mse - gt_matched_mse) / original_mse * 100)),
        "Original_Metrics": original_metrics,
        "GT_Matched_Metrics": gt_matched_metrics
    }

# 전체 평균 성능 (기존 MSE)
avg_original_mse = total_original_mse / len(all_names)
avg_gt_matched_mse = total_gt_matched_mse / len(all_names)
avg_improvement = ((avg_original_mse - avg_gt_matched_mse) / avg_original_mse * 100)

# 포괄적인 평가 지표들의 평균 계산
avg_original_metrics = {}
avg_gt_matched_metrics = {}

# 사용 가능한 지표들
metric_names = ['PSNR', 'SSIM', 'RMSE', 'MRAE', 'SAM', 'FID']

for metric in metric_names:
    original_values = []
    gt_matched_values = []
    
    for sample_name in all_names:
        if metric in performance_results[sample_name]['Original_Metrics'] and performance_results[sample_name]['Original_Metrics'][metric] is not None:
            original_values.append(performance_results[sample_name]['Original_Metrics'][metric])
        if metric in performance_results[sample_name]['GT_Matched_Metrics'] and performance_results[sample_name]['GT_Matched_Metrics'][metric] is not None:
            gt_matched_values.append(performance_results[sample_name]['GT_Matched_Metrics'][metric])
    
    if original_values:
        avg_original_metrics[metric] = float(np.mean(original_values))
    if gt_matched_values:
        avg_gt_matched_metrics[metric] = float(np.mean(gt_matched_values))

# 전체 평가 결과 저장
performance_results["Average_Original_MSE"] = float(avg_original_mse)
performance_results["Average_GT_Matched_MSE"] = float(avg_gt_matched_mse)
performance_results["Average_Improvement(%)"] = float(avg_improvement)
performance_results["Average_Original_Metrics"] = avg_original_metrics
performance_results["Average_GT_Matched_Metrics"] = avg_gt_matched_metrics
performance_results["Config"] = {
    "data_dir": args.data_dir,
    "model": args.model,
    "run_name": args.run_name,
    "save_base": args.save_base,
    "eval_base": args.eval_base,
    "gt_matching": args.gt_matching,
}
# %%
# 평가 결과 저장 (JSON)
import json

# 평가 결과 저장 경로
os.makedirs(args.eval_base, exist_ok=True)
EVAL_RESULT_FILE = os.path.join(args.eval_base, f'{args.run_name}.json')
EVAL_RESULT_FILE_IN_RUN = os.path.join(EVAL_SAVE_DIR, 'summary.json')

try:
    with open(EVAL_RESULT_FILE, 'w') as f:
        json.dump(performance_results, f, indent=2)
    with open(EVAL_RESULT_FILE_IN_RUN, 'w') as f:
        json.dump(performance_results, f, indent=2)
    print(f"평가 결과가 {EVAL_RESULT_FILE} 및 {EVAL_RESULT_FILE_IN_RUN} 에 저장되었습니다.")
except Exception as e:
    print(f"평가 결과 저장 실패: {str(e)}")

print('=' * 60)
print('Overall Performance Summary')
print('=' * 60)
print(f'Average Original MSE: {avg_original_mse:.6f}')
print(f'Average GT Matched MSE: {avg_gt_matched_mse:.6f}')
print(f'Average Improvement: {avg_improvement:.2f}%')

print('\n=== 포괄적인 평가 지표 (Original vs GT) ===')
for metric, value in avg_original_metrics.items():
    print(f'Average Original {metric}: {value:.6f}')

print('\n=== 포괄적인 평가 지표 (GT Matched vs GT) ===')
for metric, value in avg_gt_matched_metrics.items():
    print(f'Average GT Matched {metric}: {value:.6f}')

print('\n=== 개선 효과 ===')
for metric in metric_names:
    if metric in avg_original_metrics and metric in avg_gt_matched_metrics:
        if metric in ['PSNR', 'SSIM']:  # 높을수록 좋은 지표
            improvement = ((avg_gt_matched_metrics[metric] - avg_original_metrics[metric]) / avg_original_metrics[metric]) * 100
            print(f'{metric} Improvement: {improvement:+.2f}%')
        else:  # 낮을수록 좋은 지표 (RMSE, MRAE, SAM, FID)
            improvement = ((avg_original_metrics[metric] - avg_gt_matched_metrics[metric]) / avg_original_metrics[metric]) * 100
            print(f'{metric} Improvement: {improvement:+.2f}%')

print('=' * 60)
# %%
# 저장 경로 상수 정의 (exports)
SAVE_ROOT_DIR = EXPORT_DIR
os.makedirs(SAVE_ROOT_DIR, exist_ok=True)

def save_hsi_tensor_as_bmp_images(tensor, save_dir, prefix):
    """
    HSI 텐서(512, 256, 256 또는 (C, H, W))를 512개의 256x256 bmp 이미지로 저장 (grayscale)
    Args:
        tensor (torch.Tensor 또는 np.ndarray): 저장할 데이터
        save_dir (str): 저장할 폴더 경로
        prefix (str): 파일명 접두사 (예: 'gt', 'raw', 'res')
    """
    try:
        if isinstance(tensor, torch.Tensor):
            np_data = tensor.cpu().numpy()
        else:
            np_data = tensor
        # (C, H, W) 형태 보장
        if np_data.shape[0] != 512:
            raise ValueError(f"채널 수가 512가 아님: {np_data.shape}")
        os.makedirs(save_dir, exist_ok=True)
        for ch in range(512):
            img = np_data[ch]
            # 0~1 정규화 후 0~255로 변환
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img_norm = (img - img_min) / (img_max - img_min)
            else:
                img_norm = img
            img_uint8 = (img_norm * 255).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_uint8, mode='L')
            img_filename = f"{prefix}_ch{ch:03d}.bmp"
            img_path = os.path.join(save_dir, img_filename)
            # img_pil.save(img_path)
    except Exception as e:
        print(f"{prefix} 저장 실패: {save_dir} - {str(e)}")

for i in tqdm(range(len(all_names)), desc='Saving', position=0, total=len(all_names)):
    sample_name = all_names[i]
    # 각 결과별 폴더 생성
    gt_dir = os.path.join(SAVE_ROOT_DIR, f"{sample_name}_gt")
    raw_dir = os.path.join(SAVE_ROOT_DIR, f"{sample_name}_raw")
    res_dir = os.path.join(SAVE_ROOT_DIR, f"{sample_name}_res")

    # 저장
    save_hsi_tensor_as_bmp_images(hsi_tensor, gt_dir, 'gt')
    save_hsi_tensor_as_bmp_images(res[i], raw_dir, 'raw')
    save_hsi_tensor_as_bmp_images(res_gt_matched[i], res_dir, 'res')



# %%

"""
================================================================================
HSI (Hyperspectral Image) 평가 지표의 특성과 물리적 의미
================================================================================

1. PSNR (Peak Signal-to-Noise Ratio) - 피크 신호 대 잡음비
   - 물리적 의미: 원본 신호의 최대값 대비 잡음(오차)의 크기를 dB 단위로 표현
   - 특성: 높을수록 좋음 (일반적으로 20dB 이상이면 양호, 30dB 이상이면 우수)
   - HSI에서의 의미: 스펙트럼 정보의 정확도와 노이즈 수준을 종합적으로 평가
   - 계산식: 10 * log10((MAX^2) / MSE)

2. SSIM (Structural Similarity Index) - 구조적 유사성 지수
   - 물리적 의미: 인간의 시각 시스템이 인식하는 구조적 정보의 유사성을 수치화
   - 특성: 높을수록 좋음 (0~1 범위, 1에 가까울수록 완벽한 유사성)
   - HSI에서의 의미: 공간적 구조와 텍스처 정보의 보존 정도를 평가
   - 구성요소: 밝기, 대비, 구조적 유사성을 종합하여 계산

3. RMSE (Root Mean Square Error) - 평균 제곱근 오차
   - 물리적 의미: 예측값과 실제값 간의 평균적인 차이의 크기
   - 특성: 낮을수록 좋음 (0에 가까울수록 정확)
   - HSI에서의 의미: 픽셀별 스펙트럼 값의 절대적 오차를 종합적으로 평가
   - 계산식: sqrt(mean((pred - gt)^2))

4. MRAE (Mean Relative Absolute Error) - 평균 상대 절대 오차
   - 물리적 의미: 각 픽셀에서 발생하는 오차를 해당 픽셀의 실제값으로 정규화한 평균
   - 특성: 낮을수록 좋음 (상대적 오차이므로 스케일에 무관)
   - HSI에서의 의미: 스펙트럼 강도에 따른 상대적 정확도를 평가
   - 계산식: mean(|pred - gt| / |gt|)

5. SAM (Spectral Angle Mapper) - 스펙트럼 각도 매퍼
   - 물리적 의미: 두 스펙트럼 벡터 간의 각도를 라디안 단위로 측정
   - 특성: 낮을수록 좋음 (0에 가까울수록 스펙트럼 형태가 유사)
   - HSI에서의 의미: 스펙트럼의 형태와 패턴의 유사성을 평가 (강도는 무관)
   - 계산식: arccos(dot(gt, pred) / (||gt|| * ||pred||))

6. FID (Fréchet Inception Distance) - 프레셰 인셉션 거리
   - 물리적 의미: 두 확률 분포(실제 데이터와 생성 데이터) 간의 통계적 거리
   - 특성: 낮을수록 좋음 (0에 가까울수록 분포가 유사)
   - HSI에서의 의미: 생성된 HSI가 실제 HSI의 통계적 특성을 얼마나 잘 모방하는지 평가
   - 계산식: ||μ_gt - μ_pred||^2 + Tr(Σ_gt + Σ_pred - 2*sqrt(Σ_gt * Σ_pred))

================================================================================
지표별 HSI 재구성 품질 평가 관점
================================================================================

- 공간적 품질: SSIM, RMSE
- 스펙트럼 품질: SAM, MRAE, PSNR
- 전역적 품질: FID
- 노이즈 수준: PSNR, RMSE
- 구조 보존: SSIM, SAM
- 통계적 유사성: FID

================================================================================
실제 적용 시 고려사항
================================================================================

1. 데이터 특성에 따른 지표 선택:
   - 스펙트럼 분석이 중요한 경우: SAM, MRAE 우선
   - 공간적 구조가 중요한 경우: SSIM, RMSE 우선
   - 전반적 품질이 중요한 경우: PSNR 우선

2. 임계값 설정:
   - SAM: 0.1 라디안(약 5.7도) 이하가 양호

3. 지표 해석:
   - 절대적 지표: PSNR, RMSE, SAM
   - 상대적 지표: MRAE, FID
   - 비율 지표: SSIM
"""
# %%