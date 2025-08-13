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
from utils.tucker_utils import get_optimal_core_shape
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms.functional as TF
from scipy.signal import savgol_filter
%matplotlib inline
#%%
# 신경망 기반 후처리 모델 로드
def load_post_process_model(model_path, device, hsi_shape=(512, 256, 256)):
    """후처리 신경망 모델을 로드합니다."""
    from advanced_post_process_network import AdvancedPostProcessNetwork, AdvancedPostProcessTrainer
    
    model = AdvancedPostProcessNetwork(input_channels=hsi_shape[0])
    trainer = AdvancedPostProcessTrainer(model, device)
    trainer.load_model(model_path)
    
    return trainer

def post_process_hsi_with_nn(hsi: torch.Tensor, trainer, device):
    """신경망을 사용한 HSI 후처리"""
    # 입력은 이미 0-1 범위로 정규화되어 있음 (학습 시와 동일)
    
    # 배치 차원 추가
    hsi_batch = hsi.unsqueeze(0).to(device)
    
    # 신경망 추론
    trainer.model.eval()
    with torch.no_grad():
        processed_batch = trainer.model(hsi_batch)
    
    # 배치 차원 제거
    processed = processed_batch.squeeze(0).cpu()
    
    # 0-255 범위로 스케일링 (학습 시 출력과 일치)
    processed = torch.clamp(processed, 0, 1)  # 0-1 범위로 클램핑
    processed = processed * 255.0  # 0-255 범위로 변환
    
    return processed



print('=' * 60)
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/app/datas/hsi')
parser.add_argument('--model', type=str, default='/app/weights/202507301154/epoch_3000.pth')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_base', type=str, default='/app/results/202507301154') 
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.02)
parser.add_argument('--beta_schedule', type=str, default='cosine')
parser.add_argument('--num_diffusion_timesteps', type=int, default=500)
parser.add_argument('--core_shape', type=tuple, default=(64, 128, 128))
args = parser.parse_args(args=[])  # for jupyter notebook
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
# 후처리 신경망 로드
print('=' * 60)
print('Loading post-processing neural network...')
print('=' * 60)
post_process_model_path = '/app/weights/post_process_advanced/final_model.pth'
if os.path.exists(post_process_model_path):
    post_process_trainer = load_post_process_model(post_process_model_path, device, HSI_SHAPE)
    print('Advanced post-processing neural network loaded successfully!')
print('=' * 60)
# %%
print('Target dataset: ', args.data_dir)
for data in os.listdir(args.data_dir):
    print("    └", data)

test_dataset = HsiDataset(args.data_dir, HSI_SHAPE, CORE_SHAPE, no_hsi=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print('=' * 60)
print('Dataset loaded successfully!')
print('=' * 60)
# %%
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from torch.cuda.amp import autocast
from tqdm import tqdm

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
all_names = []
res = []

for batch_idx, (names, rgb, seg) in tqdm(enumerate(test_loader), desc='Testing', position=0, total=len(test_loader)):

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
            for i in range(len(re)):
                res.append(re[i])
            # names가 문자열일 수 있으므로 리스트로 변환
            if isinstance(names, str):
                names = [names]
            all_names.extend(list(names))
            # 생성 이미지 저장
            save_base = args.save_base
            os.makedirs(save_base, exist_ok=True)
            for idx, name in enumerate(names):
                save_folder_gen = f'{name}_gen'
                save_dir_gen = os.path.join(save_base, save_folder_gen)
                os.makedirs(save_dir_gen, exist_ok=True)
                for band_idx, hsi in enumerate(re[idx]): # (C, H, W)
                    plt.imsave(os.path.join(save_dir_gen, f'{save_folder_gen}_{band_idx}.bmp'), hsi, cmap='gray')
                # 원본 bmp 파일도 같은 구조로 저장
                src_dir = os.path.join(args.data_dir, name)
                save_folder_ori = name
                save_dir_ori = os.path.join(save_base, save_folder_ori)
                os.makedirs(save_dir_ori, exist_ok=True)
                hsi_files = sorted(
                    [f for f in os.listdir(src_dir) if f.endswith('.bmp')],
                    key=lambda x: int(x.split('nm')[0].split('_')[-1])
                )
                for band_idx, fn in enumerate(hsi_files):
                    img = Image.open(os.path.join(src_dir, fn)).convert('L').resize((HSI_SHAPE[2], HSI_SHAPE[1]))
                    img.save(os.path.join(save_dir_ori, f'{save_folder_ori}_{band_idx}.bmp'))
            
            raise Exception("Stop")
# %%
print('=' * 60)
print('Visualizing results...')
print('=' * 60)

print(len(res), len(all_names))
# %%
for i in range(len(all_names)):
    print(f'{i}/{len(all_names)}', all_names[i], res[i].shape)
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
    hsi_tensor = torch.from_numpy(hsi_stack)
    hsi_tensor = hsi_tensor.float()
    # #############################################################
    from sklearn.metrics import mean_squared_error
    final_hsi_tensor = torch.from_numpy(res[i])
    final_hsi_tensor = final_hsi_tensor.float()
    
    # 후처리 적용
    # final_hsi_new = post_process_hsi_with_nn(final_hsi_tensor_normalized, post_process_trainer, device)

    # print("Using neural network post-processing")
    final_hsi_new = final_hsi_tensor
    final_hsi_tensor_scaled = final_hsi_tensor
    final_hsi_new_scaled = final_hsi_new
    
    print(f"Original MSE: {mean_squared_error(final_hsi_tensor_scaled.flatten(), hsi_tensor.flatten()):.4f}")
    print(f"Post-processed MSE: {mean_squared_error(final_hsi_new_scaled.flatten(), hsi_tensor.flatten()):.4f}")

    fig, axs = plt.subplots(16, 32, figsize=(32, 16))
    fig.suptitle('Original HSI')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for idx, hsi in enumerate(hsi_tensor): # (C, H, W)
        axs[idx//32, idx%32].imshow(hsi, cmap='gray')
        axs[idx//32, idx%32].axis('off')
    plt.show()  # 자동화 실행을 위해 주석 처리

    fig, axs = plt.subplots(16, 32, figsize=(32, 16))
    fig.suptitle('Generated HSI')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for idx, hsi in enumerate(final_hsi_tensor): # (C, H, W)
        axs[idx//32, idx%32].imshow(hsi, cmap='gray')
        axs[idx//32, idx%32].axis('off')
    plt.show()  # 자동화 실행을 위해 주석 처리

    fig, axs = plt.subplots(16, 32, figsize=(32, 16))
    fig.suptitle('Post-processed HSI')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    for idx, hsi in enumerate(final_hsi_new): # (C, H, W)
        axs[idx//32, idx%32].imshow(hsi, cmap='gray')
        axs[idx//32, idx%32].axis('off')
    plt.show()  # 자동화 실행을 위해 주석 처리
# %%
import matplotlib.pyplot as plt

# 비교할 채널 인덱스 상수 정의
COMPARE_CHANNELS = [0, 127, 255, 383, 511]
NUM_ROWS = 3  # Original, Generated, Post-processed
NUM_COLS = len(COMPARE_CHANNELS)

# 각 행의 타이틀
ROW_TITLES = ['Original', 'Generated', 'Post-processed']

# 3단 비교를 위한 데이터 준비
# hsi_tensor: 원본, final_hsi_tensor: 생성, final_hsi_new: 개선
compare_data = [
    hsi_tensor,           # Original
    final_hsi_tensor,     # Generated
    final_hsi_new         # Post-processed
]

fig, axs = plt.subplots(NUM_ROWS, NUM_COLS, figsize=(NUM_COLS * 3, NUM_ROWS * 3))
fig.suptitle('HSI Comparison by Channel', fontsize=18)
fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.88, wspace=0.05, hspace=0.15)

for row in range(NUM_ROWS):
    for col, ch in enumerate(COMPARE_CHANNELS):
        ax = axs[row, col]
        # 각 데이터에서 해당 채널 이미지 추출
        try:
            img = compare_data[row][ch]
        except Exception as e:
            print(f"Error: Cannot access channel {ch} in row {row}: {e}")
            img = None
        if img is not None:
            ax.imshow(img, cmap='gray')
        ax.axis('off')
        # 첫 행에만 채널 번호 표시
        if row == 0:
            ax.set_title(f'Channel {ch}', fontsize=12)
        # 첫 열에만 행 타이틀 표시
        if col == 0:
            ax.annotate(ROW_TITLES[row], xy=(-0.25, 0.5), xycoords='axes fraction',
                        fontsize=14, ha='right', va='center', rotation=90, fontweight='bold')

plt.show()
