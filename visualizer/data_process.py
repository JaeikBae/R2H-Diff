# 필요한 라이브러리 다시 임포트
import numpy as np
import matplotlib.pyplot as plt

# 3D 시각화를 위한 라이브러리
from mpl_toolkits.mplot3d import Axes3D

# 가상의 3차원 HSI 이미지 데이터 생성 (512, 128, 128)
depth, height, width = 512, 128, 128
full_hsi_image = np.random.rand(depth, height, width)

# 특정 pos 값 선택 (예: 50)
pos = 50

# 3D HSI 데이터 큐브에서 특정 단면 시각화
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# 3D 큐브 크기 설정
x, y, z = np.meshgrid(np.arange(width+1), np.arange(height+1), np.arange(depth+1))

# 전체 3D 데이터 큐브의 외곽선
ax.voxels(np.zeros((width, height, depth)), facecolors='gray', edgecolors='gray', alpha=0.1)

# 선택한 pos 번째 밴드 (XY 평면)
selected_slice = np.zeros((width, height, depth), dtype=bool)
selected_slice[:, :, pos] = True

# 선택된 단면을 색상으로 강조
ax.voxels(selected_slice, facecolors='blue', edgecolors='black', alpha=0.6)

# 축 설정
ax.set_xlabel("Width (X)")
ax.set_ylabel("Height (Y)")
ax.set_zlabel("Spectral Bands (Z)")
ax.set_title(f"HSI Data Cube - Spectral Band {pos} (XY Slice)")

plt.savefig("./gen_images/test.png")
