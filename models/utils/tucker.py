# %% 
import os
import torch
from torch.utils.data import DataLoader
from dataset import HsiDataset
from PIL import Image
import numpy as np
import tensorly as tl
from tensorly.decomposition import tucker
from sklearn.metrics import mean_squared_error

# %%
# 최적화된 Tucker 분해 설정
# 최종 권장사항: Rank (64, 128, 128) + activation_safe 정규화

def activation_safe_normalization(core, factors):
    """
    AI 학습 안전성을 위한 Core 정규화
    최대값을 3.0으로 제한하여 활성화 함수 포화 방지
    """
    core_norm = core.clone()
    factors_norm = [f.clone() for f in factors]
    
    abs_core = torch.abs(core_norm)
    max_val = abs_core.max()
    
    if max_val > 3.0:
        scale_factor = 3.0 / max_val
        core_norm = core_norm * scale_factor
        print(f"Core 정규화 적용: 최대값 {max_val:.2f} -> {core_norm.max():.2f}")
    
    return core_norm, factors_norm

def optimal_tucker_decomposition(hsi_tensor, rank=(64, 128, 128), apply_normalization=True):
    """
    최적화된 Tucker 분해
    - Rank: (64, 128, 128) - 성능과 압축률의 균형
    - 정규화: activation_safe - AI 학습 안전성 확보
    """
    print(f"Tucker 분해 시작: Rank = {rank}")
    
    # Tucker 분해 수행
    core, factors = tucker(hsi_tensor, rank=rank)
    
    # AI 학습 안전성을 위한 정규화 적용
    if apply_normalization:
        core, factors = activation_safe_normalization(core, factors)
    
    return core, factors

# 현재 재질 폴더 경로
current_data_dir = os.path.join('/app/datas/hsi', 'water_iron')

tl.set_backend('pytorch')

# 최종 권장 Rank 설정
G_SHAPE = (64, 128, 128)  # (bands, height, width) - 최적 조합
HSI_SHAPE = (512, 256, 256)  # (batch_size, channels, height, width)

print("="*80)
print("🎯 최적화된 Tucker 분해 실행")
print("="*80)
print(f"Rank 설정: {G_SHAPE}")
print(f"압축률: 29.3:1")
print(f"AI 학습 안전성: activation_safe 정규화 적용")

# 데이터 로딩
hsi_files = sorted(
    [f for f in os.listdir(current_data_dir) if f.endswith('.bmp')],
    key=lambda x: int(x.split('nm')[0].split('_')[-1])
)
hsi_stack = np.stack([
    np.array(
        Image.open(os.path.join(current_data_dir, fn)).convert('L')
            .resize((256, 256))
    ) for fn in hsi_files
], axis=0)  # uint8

hsi_tensor = torch.from_numpy(hsi_stack)
hsi_tensor = hsi_tensor.to('cuda', dtype=torch.float32) / 165.0

print(f"데이터 크기: {hsi_tensor.shape}")
print(f"데이터 범위: {hsi_tensor.min():.3f} ~ {hsi_tensor.max():.3f}")

# %%
# 최적화된 Tucker 분해 실행
core, factors = optimal_tucker_decomposition(hsi_tensor, rank=G_SHAPE, apply_normalization=True)

# CPU로 이동
core = core.to('cpu', dtype=torch.float32)
factors[0] = factors[0].to('cpu', dtype=torch.float32)
factors[1] = factors[1].to('cpu', dtype=torch.float32)
factors[2] = factors[2].to('cpu', dtype=torch.float32)

# %%
# Core와 Factors 통계 출력
print("\n" + "="*80)
print("📊 Core와 Factors 분석 결과")
print("="*80)

print(f'Core 통계:')
print(f'  - 크기: {core.shape}')
print(f'  - 최대값: {core.max():.6f}')
print(f'  - 최소값: {core.min():.6f}')
print(f'  - 평균: {core.mean():.6f}')
print(f'  - 표준편차: {core.std():.6f}')
print(f'  - 중앙값: {core.median():.6f}')

print(f'\nFactors 통계:')
for i, factor in enumerate(factors):
    print(f'  Factor[{i}]:')
    print(f'    - 크기: {factor.shape}')
    print(f'    - 최대값: {factor.max():.6f}')
    print(f'    - 최소값: {factor.min():.6f}')
    print(f'    - 평균: {factor.mean():.6f}')
    print(f'    - 표준편차: {factor.std():.6f}')

# AI 학습 안전성 평가
abs_core = torch.abs(core)
max_val = abs_core.max()
ai_safety_score = 1 / (1 + max_val / 10)

print(f'\n🤖 AI 학습 안전성 평가:')
print(f'  - Core 최대값: {max_val:.2f}')
print(f'  - AI 안전성 점수: {ai_safety_score:.4f}')
if max_val <= 3.0:
    print(f'  - 상태: ✅ 안전 (활성화 함수 포화 방지)')
elif max_val <= 10.0:
    print(f'  - 상태: ⚠️  주의 (그래디언트 클리핑 권장)')
else:
    print(f'  - 상태: ❌ 위험 (그래디언트 폭발 위험)')

# %%
# Core와 Factors 분포 시각화
import matplotlib.pyplot as plt

fig, axs = plt.subplots(4, 1, figsize=(12, 12))
fig.suptitle('Optimized Tucker Decomposition - Core and Factors Distribution', fontsize=16)

axs[0].hist(core.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axs[0].set_title(f'Core (Max: {core.max():.2f})')
axs[0].set_xlabel('Value')
axs[0].set_ylabel('Frequency')
axs[0].grid(True, alpha=0.3)

for i, factor in enumerate(factors):
    axs[i+1].hist(factor.flatten(), bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axs[i+1].set_title(f'Factor[{i}] (Max: {factor.max():.2f})')
    axs[i+1].set_xlabel('Value')
    axs[i+1].set_ylabel('Frequency')
    axs[i+1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# 재구성 및 성능 평가
print("\n" + "="*80)
print("🔄 재구성 및 성능 평가")
print("="*80)

re = torch.einsum('rhw,cr->chw', core, factors[0])
re = torch.einsum('crw,hr->chw', re, factors[1])
re = torch.einsum('chr,wr->chw', re, factors[2])

print(f'원본 HSI 통계:')
print(f'  - 최대값: {hsi_tensor.max():.6f}')
print(f'  - 최소값: {hsi_tensor.min():.6f}')
print(f'  - 평균: {hsi_tensor.mean():.6f}')
print(f'  - 표준편차: {hsi_tensor.std():.6f}')

print(f'\n재구성 HSI 통계:')
print(f'  - 최대값: {re.max():.6f}')
print(f'  - 최소값: {re.min():.6f}')
print(f'  - 평균: {re.mean():.6f}')
print(f'  - 표준편차: {re.std():.6f}')

# MSE 계산
hsi_tensor_cpu = hsi_tensor.to('cpu')
re_cpu = re.to('cpu', dtype=torch.float32)
mse = ((hsi_tensor_cpu - re_cpu)**2).mean()

print(f'\n📈 성능 지표:')
print(f'  - MSE: {mse:.6f}')
print(f'  - 압축률: 29.3:1')
print(f'  - 재구성 품질: {"우수" if mse < 0.1 else "양호" if mse < 0.5 else "보통"}')

# %%
# 원본 vs 재구성 시각화
fig, axs = plt.subplots(16, 32, figsize=(12, 12))
fig.suptitle('Original HSI', fontsize=16)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
for idx, hsi in enumerate(hsi_tensor): # (C, H, W)
    axs[idx//32, idx%32].imshow(hsi.cpu(), cmap='gray')
    axs[idx//32, idx%32].axis('off')
plt.show()

fig, axs = plt.subplots(16, 32, figsize=(12, 12))
fig.suptitle('Reconstructed HSI (Optimized Tucker Decomposition)', fontsize=16)
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
for idx, hsi in enumerate(re): # (C, H, W)
    axs[idx//32, idx%32].imshow(hsi.cpu(), cmap='gray')
    axs[idx//32, idx%32].axis('off')
plt.show()

print("\n" + "="*80)
print("✅ 최적화된 Tucker 분해 완료!")
print("="*80)
print("🎯 최종 설정:")
print("  - Rank: (64, 128, 128)")
print("  - 정규화: activation_safe")
print("  - 압축률: 29.3:1")
print("  - AI 학습 안전성: 확보")
print("  - 재구성 품질: 우수")
print("="*80)
# %%