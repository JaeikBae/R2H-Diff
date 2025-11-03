import os

from PIL import Image, ImageDraw
import numpy as np
import tensorly as tl
import torch
import torch.utils.data as data
from tqdm import tqdm
from .tucker_utils import optimal_tucker_decomposition

def create_single_channel_mask(
    label_path: str,
    image_size: tuple[int, int],
) -> Image.Image:
    """
    label_path: YOLOv8(seg) 스타일 라벨 파일 경로
    image_size: (width, height)
    output_path: 결과 마스크를 저장할 경로 (None이면 저장하지 않음)
    """
    w, h = image_size
    mode = "L"
    mask = Image.new(mode, (w, h), 0)   # 배경값 0
    draw = ImageDraw.Draw(mask)

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_idx = int(parts[0])
            coords = list(map(float, parts[1:]))

            # 정규화된 좌표 → 픽셀 좌표
            points = [
                (coords[i] * w, coords[i+1] * h)
                for i in range(0, len(coords), 2)
            ]
            # 폴리곤 내부를 cls_idx+1 로 채움
            draw.polygon(points, fill=cls_idx+1)
    return mask

class HsiDataset(data.Dataset):
    def __init__(self, path, HSI_shape, G_shape, disable_tqdm=False, is_eval=False, no_hsi=False):
        '''
        is_eval: only returns rgb and seg. Reduces memory usage and speeds up inference.
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tl.set_backend('pytorch')
        self.H, self.W = HSI_shape[1], HSI_shape[2] # (height, width)
        self.G_shape = G_shape
        self.data = []
        self.is_eval = is_eval
        self.no_hsi = no_hsi

        folders = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        for folder in tqdm(folders, desc='Loading dataset', disable=disable_tqdm):
            folder_path = os.path.join(path, folder)
            if not self.no_hsi:
                # 1) HSI 로드 & 스택 → (512, H, W)
                hsi_files = sorted(
                    [f for f in os.listdir(folder_path) if f.endswith('.bmp')],
                    key=lambda x: int(x.split('nm')[0].split('_')[-1])
                )
                hsi_stack = np.stack([
                    np.array(
                        Image.open(os.path.join(folder_path, fn)).convert('L')
                            .resize((self.W, self.H))
                    ) for fn in hsi_files
                ], axis=0)  # uint8
                hsi_tensor = torch.from_numpy(hsi_stack)
                hsi_tensor = hsi_tensor.float()
                hsi_tensor = hsi_tensor.div_(165.0)  # torch.FloatTensor
                hsi_tensor = hsi_tensor.to(device)
                # 최적화된 Tucker 분해 적용
                core, factors = optimal_tucker_decomposition(hsi_tensor, rank=self.G_shape, apply_normalization=True)
                core = core.to('cpu') # this is not error. but warning
                factors = [factor.to('cpu') for factor in factors]
            # 2) RGB 로드 → (3, H, W)
            rgb_name = next(f for f in os.listdir(folder_path) if f.endswith('.jpg'))
            rgb_np = np.array(
                Image.open(os.path.join(folder_path, rgb_name)).convert('RGB')
                    .resize((self.W, self.H))
            )
            rgb_tensor = torch.from_numpy(rgb_np.transpose(2,0,1)).float().div_(255.0)

            # 3) Mask 생성 → (H, W)
            seg_name = next(f for f in os.listdir(folder_path) if f.endswith('.txt'))
            seg_mask = create_single_channel_mask(
                os.path.join(folder_path, seg_name),
                image_size=(self.W, self.H)
            )
            seg_np = np.array(seg_mask)
            seg_tensor = torch.from_numpy(seg_np).float().div_(9.0)  # (H, W)
            seg_tensor = seg_tensor.unsqueeze(0)
            self.data.append({  
                'folder': folder,
                'core': core if not self.no_hsi else None,            # G_SHAPE  
                'f_0': factors[0] if not self.no_hsi else None,     # (512, G_SHAPE[0])
                'f_1': factors[1] if not self.no_hsi else None,     # (H, G_SHAPE[1])
                'f_2': factors[2] if not self.no_hsi else None,     # (W, G_SHAPE[2])
                'rgb': rgb_tensor,      # (3, H, W)
                'seg': seg_tensor       # (1, H, W)
            })
        self.num_folders = len(self.data)

    def __len__(self):
        return self.num_folders

    def __getitem__(self, idx):
        rec = self.data[idx]
        if self.no_hsi:
            return rec['folder'], rec['rgb'], rec['seg']
        if self.is_eval:
            return rec['folder'], rec['core'], rec['f_0'], rec['f_1'], rec['f_2'], rec['rgb'], rec['seg']
        return rec['core'], rec['f_0'], rec['f_1'], rec['f_2'], rec['rgb'], rec['seg']