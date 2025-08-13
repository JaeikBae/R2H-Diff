import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

class PostProcessDataset(Dataset):
    """
    생성된 HSI와 원본 HSI 쌍을 로드하는 데이터셋
    """
    def __init__(self, generated_dir, original_dir, hsi_shape=(512, 256, 256)):
        self.generated_dir = generated_dir
        self.original_dir = original_dir
        self.hsi_shape = hsi_shape
        self.samples = []
        
        # 생성된 이미지 폴더들 찾기
        generated_folders = [d for d in os.listdir(generated_dir) if d.endswith('_gen')]
        
        for folder in generated_folders:
            original_folder = folder.replace('_gen', '')
            generated_path = os.path.join(generated_dir, folder)
            original_path = os.path.join(original_dir, original_folder)
            
            if os.path.exists(original_path):
                self.samples.append({
                    'generated_path': generated_path,
                    'original_path': original_path,
                    'name': original_folder
                })
    
    def __len__(self):
        return len(self.samples)
    
    def load_hsi_from_folder(self, folder_path):
        """폴더에서 HSI 이미지들을 로드하여 텐서로 변환"""
        hsi_files = []
        for f in os.listdir(folder_path):
            if f.endswith('.bmp'):
                try:
                    # 생성된 파일: blockwood_gen_0.bmp 형식
                    if '_gen_' in f:
                        band_num = int(f.split('_gen_')[-1].split('.')[0])
                        hsi_files.append((f, band_num))
                    elif f.split('_')[-1].split('.')[0].isdigit():
                        band_num = int(f.split('_')[-1].split('.')[0])
                        hsi_files.append((f, band_num))
                    else:
                        # 기타 형식: 숫자로 끝나는 경우
                        band_num = int(f.split('_')[-1].split('.')[0])
                        hsi_files.append((f, band_num))
                except:
                    continue
        
        # 밴드 번호로 정렬
        hsi_files.sort(key=lambda x: x[1])
        hsi_files = [f[0] for f in hsi_files]
        
        hsi_tensor = torch.from_numpy(np.stack([
                    np.array(
                        Image.open(os.path.join(folder_path, fn)).convert('L')
                            .resize((self.hsi_shape[2], self.hsi_shape[1]))
                    ) for fn in hsi_files
                ], axis=0)).float()
        print(hsi_tensor.max(), hsi_tensor.min(), hsi_tensor.shape)
        hsi_tensor = hsi_tensor / 255.0  # 정규화
        print(hsi_tensor.max(), hsi_tensor.min(), hsi_tensor.shape)
        raise Exception("Stop")
        return hsi_tensor  # (C, H, W)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 생성된 HSI 로드
        generated_hsi = self.load_hsi_from_folder(sample['generated_path'])
        
        # 원본 HSI 로드
        original_hsi = self.load_hsi_from_folder(sample['original_path'])
        
        return {
            'generated': generated_hsi,
            'original': original_hsi,
            'name': sample['name']
        }

class PostProcessNetwork(nn.Module):
    """
    생성된 HSI를 원본 HSI로 후처리하는 신경망
    """
    def __init__(self, input_channels=512, hidden_channels=256):
        super(PostProcessNetwork, self).__init__()
        
        # 인코더: 생성된 HSI를 특징으로 변환
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels//2),
            nn.ReLU(inplace=True),
        )
        
        # 디코더: 특징을 원본 HSI로 변환
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels//2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # 0-1 범위로 출력
        )
        
        # 잔차 연결을 위한 1x1 컨볼루션
        self.residual_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)
        
    def forward(self, x):
        # 잔차 연결
        residual = self.residual_conv(x)
        
        # 인코더-디코더 통과
        features = self.encoder(x)
        output = self.decoder(features)
        
        # 잔차 연결 추가
        output = output + residual
        
        return output

class PostProcessTrainer:
    """
    후처리 신경망 학습을 위한 클래스
    """
    def __init__(self, model, device, lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc='Training'):
            generated = batch['generated'].to(self.device)
            original = batch['original'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # 순전파
            output = self.model(generated)
            loss = self.criterion(output, original)
            
            # 역전파
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        total_psnr = 0
        total_ssim = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                generated = batch['generated'].to(self.device)
                original = batch['original'].to(self.device)
                
                output = self.model(generated)
                loss = self.criterion(output, original)
                total_loss += loss.item()
                
                # PSNR과 SSIM 계산
                for i in range(output.shape[0]):
                    pred_np = output[i].cpu().numpy()
                    target_np = original[i].cpu().numpy()
                    
                    # 각 밴드별로 PSNR 계산
                    band_psnr = 0
                    for band in range(pred_np.shape[0]):
                        band_psnr += psnr(target_np[band], pred_np[band], data_range=1.0)
                    total_psnr += band_psnr / pred_np.shape[0]
                    
                    # SSIM 계산 (첫 번째 밴드 사용)
                    total_ssim += ssim(target_np[0], pred_np[0], data_range=1.0)
        
        avg_loss = total_loss / len(dataloader)
        avg_psnr = total_psnr / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        
        return avg_loss, avg_psnr, avg_ssim
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train_post_process_network(generated_dir, original_dir, save_dir, 
                              hsi_shape=(512, 256, 256), batch_size=2, 
                              num_epochs=100, lr=1e-4):
    """
    후처리 신경망 학습 함수
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 데이터셋 생성
    dataset = PostProcessDataset(generated_dir, original_dir, hsi_shape)
    print(f'Dataset size: {len(dataset)}')
    
    # 전체 데이터를 학습에 사용 (검증 분할 없음)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = None  # 검증 데이터셋 없음
    
    # 모델 생성
    model = PostProcessNetwork(input_channels=hsi_shape[0])
    trainer = PostProcessTrainer(model, device, lr=lr)
    
    # 학습 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 학습 루프
    train_losses = []
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        
        # 학습
        train_loss = trainer.train_epoch(train_loader)
        train_losses.append(train_loss)
        
        # 스케줄러 업데이트 (학습 손실 사용)
        trainer.scheduler.step(train_loss)
        
        print(f'Train Loss: {train_loss:.6f}')
        
        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            trainer.save_model(os.path.join(save_dir, f'epoch_{epoch+1}.pth'))
    
    # 최종 모델 저장
    trainer.save_model(os.path.join(save_dir, 'final_model.pth'))
    
    # 학습 곡선 플롯
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    # plt.show()  # 자동화 실행을 위해 주석 처리
    
    return trainer

def inference_post_process_network(model_path, generated_dir, output_dir, 
                                  hsi_shape=(512, 256, 256), batch_size=1):
    """
    학습된 후처리 신경망으로 추론 수행
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 로드
    model = PostProcessNetwork(input_channels=hsi_shape[0])
    trainer = PostProcessTrainer(model, device)
    trainer.load_model(model_path)
    
    # 데이터셋 생성
    dataset = PostProcessDataset(generated_dir, generated_dir, hsi_shape)  # 원본 디렉토리는 임시로 사용
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    trainer.model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Inference'):
            generated = batch['generated'].to(device)
            names = batch['name']
            
            # 후처리 수행
            output = trainer.model(generated)
            
            # 결과 저장
            for i, name in enumerate(names):
                save_folder = os.path.join(output_dir, f'{name}_postprocessed')
                os.makedirs(save_folder, exist_ok=True)
                
                # 각 밴드별로 저장
                for band_idx in range(output.shape[1]):
                    band_img = output[i, band_idx].cpu().numpy()
                    band_img = (band_img * 255).astype(np.uint8)
                    
                    plt.imsave(
                        os.path.join(save_folder, f'{name}_postprocessed_{band_idx}.bmp'),
                        band_img, cmap='gray'
                    )
    
    print(f'Inference completed! Results saved to {output_dir}')

def compare_results(original_dir, generated_dir, postprocessed_dir, sample_name):
    """
    원본, 생성, 후처리 결과 비교
    """
    def load_hsi_from_folder(folder_path):
        hsi_files = sorted(
            [f for f in os.listdir(folder_path) if f.endswith('.bmp')],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        
        hsi_stack = []
        for fn in hsi_files:
            img = Image.open(os.path.join(folder_path, fn)).convert('L')
            hsi_stack.append(np.array(img))
        
        return np.stack(hsi_stack, axis=0)
    
    # 각 결과 로드
    original_path = os.path.join(original_dir, sample_name)
    generated_path = os.path.join(generated_dir, f'{sample_name}_gen')
    postprocessed_path = os.path.join(postprocessed_dir, f'{sample_name}_postprocessed')
    
    original_hsi = load_hsi_from_folder(original_path)
    generated_hsi = load_hsi_from_folder(generated_path)
    postprocessed_hsi = load_hsi_from_folder(postprocessed_path)
    
    # 메트릭 계산
    def calculate_metrics(pred, target):
        mse = mean_squared_error(pred.flatten(), target.flatten())
        psnr_val = psnr(target, pred, data_range=255)
        ssim_val = ssim(target[0], pred[0], data_range=255)
        return mse, psnr_val, ssim_val
    
    gen_mse, gen_psnr, gen_ssim = calculate_metrics(generated_hsi, original_hsi)
    post_mse, post_psnr, post_ssim = calculate_metrics(postprocessed_hsi, original_hsi)
    
    print(f'Results for {sample_name}:')
    print(f'Generated - MSE: {gen_mse:.4f}, PSNR: {gen_psnr:.2f} dB, SSIM: {gen_ssim:.4f}')
    print(f'Postprocessed - MSE: {post_mse:.4f}, PSNR: {post_psnr:.2f} dB, SSIM: {post_ssim:.4f}')
    
    # 시각화
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # 중간 밴드들 선택
    mid_bands = [original_hsi.shape[0]//4, original_hsi.shape[0]//2, 3*original_hsi.shape[0]//4, original_hsi.shape[0]-1]
    
    for i, band_idx in enumerate(mid_bands):
        axes[0, i].imshow(original_hsi[band_idx], cmap='gray')
        axes[0, i].set_title(f'Original Band {band_idx}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(generated_hsi[band_idx], cmap='gray')
        axes[1, i].set_title(f'Generated Band {band_idx}')
        axes[1, i].axis('off')
        
        axes[2, i].imshow(postprocessed_hsi[band_idx], cmap='gray')
        axes[2, i].set_title(f'Postprocessed Band {band_idx}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    # plt.show()  # 자동화 실행을 위해 주석 처리
    
    return {
        'generated': {'mse': gen_mse, 'psnr': gen_psnr, 'ssim': gen_ssim},
        'postprocessed': {'mse': post_mse, 'psnr': post_psnr, 'ssim': post_ssim}
    } 