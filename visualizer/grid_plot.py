# %%
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Union

# 이미지 폴더에서 이미지 파일을 로드하는 함수
def load_images_from_directory(directory_path: str, pattern: str = '*.png') -> List[np.ndarray]:
    image_files = glob.glob(os.path.join(directory_path, pattern))
    image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 예: image_0.png → 정렬

    images = []
    for img_path in image_files:
        try:
            img = np.array(Image.open(img_path))
            images.append(img)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
    return images

# 여러 장의 이미지를 격자(grid) 형태로 출력하는 함수
def show_images_grid(images, rows=10, cols=5, figsize=(30, 16)):
    assert len(images) <= rows * cols, f"이미지 수는 최대 {rows*cols}개입니다."
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            # idx 표시 (왼쪽 위에 흰색으로)
            ax.text(0.01, 0.9, f'{idx}', ha='left', va='top', transform=ax.transAxes, fontsize=48, color='white')
            img = images[idx]
            # PIL.Image 객체일 경우 numpy 배열로 변환
            if not hasattr(img, 'shape'):
                img = np.array(img)
            # 그레이스케일 (2D 배열)
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            # 싱글 채널 (H, W, 1)
            elif img.ndim == 3 and img.shape[2] == 1:
                ax.imshow(img.squeeze(-1), cmap='gray')
            # 컬러 (H, W, 3)
            else:
                ax.imshow(img)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# %%
# 사용 예시
images = load_images_from_directory('../models/gen_images')
print(len(images))
# %%
show_images_grid(images)              # 여러 이미지 격자 출력

# %%
import torch.nn as nn
import torch
# show spectrum
def plot_spectrum(img1, img2, x=64):
    # 이미지 중 특정 픽셀 좌표의 스펙트럼 시그니처 출력
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    print(img1.shape, img2.shape)
    spectrum1 = img1[x, :]
    spectrum2 = img2[x, :]
    ax[0].plot(spectrum1)
    ax[1].plot(spectrum2)
    print(spectrum1.shape, spectrum2.shape)
    spectrum1 = torch.from_numpy(spectrum1).unsqueeze(0) / 255.0
    spectrum2 = torch.from_numpy(spectrum2).unsqueeze(0) / 255.0
    print(nn.functional.mse_loss(spectrum1, spectrum2))
    plt.show()

# %%
x = np.load('/app/models/gen_images/x_1.npy')[0][0] * 255
plot_spectrum(np.array(images[-1]), x)
# %%
