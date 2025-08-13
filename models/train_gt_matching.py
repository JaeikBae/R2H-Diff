import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import time
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from large_gt_matching_model import LargeGTMatchingModel
from torch.cuda.amp import GradScaler, autocast

# 상수 정의
TIME = time.strftime("%Y%m%d%H%M")
DATA_DIR = "/app/post_train"
WEIGHTS_DIR = f"/app/weights/residual_{TIME}"
RESULTS_DIR = f"/app/results/residual_{TIME}"
VISUALIZATION_DIR = f"/app/results/visualizations/residual_{TIME}"  # 시각화 결과 저장 디렉토리

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

BATCH_SIZE = 16  # 멀티 GPU에 최적화 (GPU당 2개씩)
LEARNING_RATE = 5e-5  # 더 낮은 학습률로 시작
NUM_EPOCHS = 10000
SAVE_INTERVAL = 50
VISUALIZATION_INTERVAL = 10  # 시각화 간격 (에포크)
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# 시드 고정 함수 정의 및 적용
def setSeed(seed):
    """
    시드 고정을 통해 재현성을 확보하는 함수
    Args:
        seed (int): 사용할 시드 값
    """
    try:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
    except Exception as e:
        print(f"시드 고정 오류: {e}")

setSeed(SEED)


# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{RESULTS_DIR}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GTMatchingDataset(Dataset):
    """GT Matching을 위한 데이터셋 클래스"""
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # 데이터 파일 목록 생성
        self.data_files = []
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        # GT가 아닌 파일들만 수집
        for file_path in npy_files:
            if not file_path.endswith("_gt.npy"):
                gt_file = file_path.replace(".npy", "_gt.npy")
                if os.path.exists(gt_file):
                    self.data_files.append({
                        'data': file_path,
                        'gt': gt_file
                    })
        
        logger.info(f"데이터셋 로드 완료: {len(self.data_files)}개 파일")
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        # 데이터 로드
        data_path = self.data_files[idx]['data']
        gt_path = self.data_files[idx]['gt']
        
        data = np.load(data_path).astype(np.float16)
        gt_data = np.load(gt_path).astype(np.float16)
        # 텐서로 변환 (C, H, W) 형태
        if len(data.shape) == 3:
            data = torch.from_numpy(data).float()
            gt_data = torch.from_numpy(gt_data).float()
        else:
            raise ValueError(f"예상치 못한 데이터 형태: {data.shape}")
        # 데이터는 원본 dtype 유지 (mixed precision에서 자동 처리)
        
        if self.transform:
            data = self.transform(data)
            gt_data = self.transform(gt_data)
        return data_path, data, gt_data


class VisualizationHelper:
    """학습 중간 결과 시각화를 위한 헬퍼 클래스"""
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_comparison(self, input_data, output_data, gt_data, epoch, batch_idx=0, sample_idx=0):
        """
        입력/출력/원본을 비교하여 시각화 (5개 채널, 5x3 형태)
        
        Args:
            input_data: 모델 입력 데이터 (B, C, H, W)
            output_data: 모델 출력 데이터 (B, C, H, W)
            gt_data: Ground Truth 데이터 (B, C, H, W)
            epoch: 현재 에포크
            batch_idx: 배치 인덱스
            sample_idx: 샘플 인덱스
        """
        try:
            # CPU로 이동하고 numpy로 변환
            input_np = input_data[batch_idx].cpu().numpy()  # (C, H, W)
            output_np = output_data[batch_idx].cpu().numpy()  # (C, H, W)
            gt_np = gt_data[batch_idx].cpu().numpy()  # (C, H, W)
            
            # 5개 채널 선택 (0, 127, 255, 383, 511)
            channels = [0, 127, 255, 383, 511]
            
            # 5x3 형태로 시각화 (5개 채널 x 3개 타입)
            fig, axes = plt.subplots(5, 3, figsize=(12, 20))
            
            for i, ch in enumerate(channels):
                # 입력 데이터
                axes[i, 0].imshow(input_np[ch], cmap='gray', vmin=0, vmax=1)
                axes[i, 0].set_title(f'Input Ch{ch} (Epoch {epoch})')
                axes[i, 0].axis('off')
                
                # 출력 데이터
                axes[i, 1].imshow(output_np[ch], cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title(f'Output Ch{ch} (Epoch {epoch})')
                axes[i, 1].axis('off')
                
                # Ground Truth
                axes[i, 2].imshow(gt_np[ch], cmap='gray', vmin=0, vmax=1)
                axes[i, 2].set_title(f'GT Ch{ch} (Epoch {epoch})')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f'comparison_epoch_{epoch:03d}_batch_{batch_idx:02d}_sample_{sample_idx:02d}.png'
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"시각화 저장: {filename}")
            
        except Exception as e:
            logger.error(f"시각화 오류: {e}")
    
    def visualize_spectrum_comparison(self, input_data, output_data, gt_data, epoch, batch_idx=0, sample_idx=0):
        """
        스펙트럼 비교 시각화 (특정 픽셀의 스펙트럼)
        """
        try:
            # CPU로 이동하고 numpy로 변환
            input_np = input_data[batch_idx, :, sample_idx, sample_idx].cpu().numpy()
            output_np = output_data[batch_idx, :, sample_idx, sample_idx].cpu().numpy()
            gt_np = gt_data[batch_idx, :, sample_idx, sample_idx].cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # 스펙트럼 비교
            wavelengths = np.arange(512)  # 512개 채널
            
            axes[0].plot(wavelengths, input_np, 'b-', alpha=0.7, label='Input')
            axes[0].plot(wavelengths, output_np, 'r-', alpha=0.7, label='Output')
            axes[0].plot(wavelengths, gt_np, 'g-', alpha=0.7, label='Ground Truth')
            axes[0].set_xlabel('Channel')
            axes[0].set_ylabel('Intensity')
            axes[0].set_title(f'Spectrum Comparison (Epoch {epoch})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # 차이 분석
            input_diff = np.abs(input_np - gt_np)
            output_diff = np.abs(output_np - gt_np)
            
            axes[1].plot(wavelengths, input_diff, 'b-', alpha=0.7, label='Input Error')
            axes[1].plot(wavelengths, output_diff, 'r-', alpha=0.7, label='Output Error')
            axes[1].set_xlabel('Channel')
            axes[1].set_ylabel('Absolute Error')
            axes[1].set_title(f'Error Analysis (Epoch {epoch})')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 파일 저장
            filename = f'spectrum_epoch_{epoch:03d}_batch_{batch_idx:02d}_sample_{sample_idx:02d}.png'
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"스펙트럼 시각화 저장: {filename}")
            
        except Exception as e:
            logger.error(f"스펙트럼 시각화 오류: {e}")


class GTMatchingTrainer:
    """GT Matching 모델 학습 클래스"""
    
    def __init__(self, model, device, learning_rate=5e-5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # 옵티마이저 및 손실 함수 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5,
            eps=1e-8  # 더 안정적인 epsilon 값
        )
        
        # Cosine Annealing 스케줄러
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=NUM_EPOCHS,
            eta_min=1e-7
        )
        
        # 손실 함수 (Huber Loss - 더 안정적)
        self.criterion = nn.HuberLoss(delta=0.1)
        
        # Mixed Precision Training을 위한 GradScaler
        self.scaler = GradScaler()
        
        # 학습 기록
        self.train_losses = []
        self.learning_rates = []
        
        # 최고 성능 모델 추적 (전체 데이터 학습이므로 학습 손실 기준)
        self.best_train_loss = float('inf')
        self.best_epoch = 0
        
        # 시각화 헬퍼
        self.visualizer = VisualizationHelper(VISUALIZATION_DIR)
        
        logger.info(f"학습기 초기화 완료 (디바이스: {device})")
        
    def train_epoch(self, train_loader, epoch):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        total_input_gt_mse = 0.0  # 입력-GT MSE
        total_output_gt_mse = 0.0  # 출력-GT MSE
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"{epoch}/{NUM_EPOCHS}")
        
        for batch_idx, (data_path, data, gt_data) in enumerate(progress_bar):
            try:
                # 데이터를 디바이스로 이동
                data = data.to(self.device, non_blocking=True)
                gt_data = gt_data.to(self.device, non_blocking=True)
                
                # 그래디언트 초기화
                self.optimizer.zero_grad()
                
                # Mixed Precision Training
                with autocast():
                    # 순전파
                    output = self.model(data)
                    # 손실 계산 (Huber Loss)
                    loss = self.criterion(output, gt_data)
                    
                    # MSE 계산 (입력-GT vs 출력-GT 비교)
                    input_gt_mse = torch.mean((data - gt_data) ** 2).item()
                    output_gt_mse = torch.mean((output - gt_data) ** 2).item()
                
                # 역전파 (GradScaler 사용)
                self.scaler.scale(loss).backward()
                
                # 그래디언트 클리핑
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                # 옵티마이저 스텝
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                
                total_loss += loss.item()
                total_input_gt_mse += input_gt_mse
                total_output_gt_mse += output_gt_mse
                
                # 시각화 (주기적으로)
                if (epoch + 1) % VISUALIZATION_INTERVAL == 0 and batch_idx == 0:
                    with torch.no_grad():
                        self.visualizer.visualize_comparison(
                            data, output, gt_data, epoch + 1, batch_idx, 0
                        )
                        self.visualizer.visualize_spectrum_comparison(
                            data, output, gt_data, epoch + 1, batch_idx, 0
                        )
                
                # 진행률 업데이트
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg': f'{total_loss/(batch_idx+1):.6f}',
                    'Input-GT MSE': f'{input_gt_mse:.6f}',
                    'Output-GT MSE': f'{output_gt_mse:.6f}',
                    'Improvement': f'{((input_gt_mse - output_gt_mse) / input_gt_mse * 100):.2f}%'
                })
                
            except Exception as e:
                logger.error(f"배치 학습 오류 (배치 {batch_idx}): {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        self.train_losses.append(avg_loss)
        
        # 최고 성능 모델 업데이트 (전체 데이터 학습이므로)
        if avg_loss < self.best_train_loss:
            self.best_train_loss = avg_loss
            self.best_epoch = epoch
        
        return avg_loss
    

    
    def save_checkpoint(self, epoch, is_best=False):
        """체크포인트 저장"""
        try:
            os.makedirs(WEIGHTS_DIR, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'train_losses': self.train_losses,
                'best_train_loss': self.best_train_loss,
                'best_epoch': self.best_epoch,
                'learning_rate': self.learning_rate
            }
            
            # 일반 체크포인트 저장 (대형 모델)
            checkpoint_path = os.path.join(WEIGHTS_DIR, f'large_gt_matching_epoch_{epoch}.pth')
            torch.save(checkpoint, checkpoint_path)
            
            # 최고 성능 모델 저장 (대형 모델)
            if is_best:
                best_path = os.path.join(WEIGHTS_DIR, 'large_gt_matching_best.pth')
                torch.save(checkpoint, best_path)
                logger.info(f"최고 성능 모델 저장: {best_path}")
            
        except Exception as e:
            logger.error(f"체크포인트 저장 오류: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        """체크포인트 로드"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # DataParallel 모델 고려
            state_dict = checkpoint['model_state_dict']
            
            # 현재 모델이 DataParallel인지 확인
            if isinstance(self.model, nn.DataParallel):
                # 체크포인트에 'module.' 접두사가 없으면 추가
                if not any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = f'module.{key}'
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
            else:
                # 체크포인트에 'module.' 접두사가 있으면 제거
                if any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            new_key = key[7:]  # 'module.' 제거
                        else:
                            new_key = key
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
            
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.train_losses = checkpoint.get('train_losses', [])
            self.best_train_loss = checkpoint.get('best_train_loss', float('inf'))
            self.best_epoch = checkpoint.get('best_epoch', 0)
            
            logger.info(f"체크포인트 로드 완료: {checkpoint_path}")
            return checkpoint['epoch']
            
        except Exception as e:
            logger.error(f"체크포인트 로드 오류: {e}")
            return 0
    
    def plot_training_curves(self):
        """학습 곡선 플롯"""
        try:
            os.makedirs(RESULTS_DIR, exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # 손실 곡선
            ax1.plot(self.train_losses, label='Train Loss', color='blue')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss (Full Dataset)')
            ax1.legend()
            ax1.grid(True)
            
            # 학습률 곡선
            if self.learning_rates:
                ax2.plot(self.learning_rates, label='Learning Rate', color='green')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate Schedule')
                ax2.legend()
                ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, 'training_curves.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("학습 곡선 저장 완료")
            
        except Exception as e:
            logger.error(f"학습 곡선 저장 오류: {e}")


def main():
    """메인 학습 함수"""
    logger.info("GT Matching 모델 학습 시작")
    
    # 디렉토리 생성
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # GPU 정보 출력
    if torch.cuda.is_available():
        logger.info(f"멀티 GPU 설정: {NUM_GPUS}개 GPU 사용")
        for i in range(NUM_GPUS):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(NUM_GPUS)) / (1024**3)
        logger.info(f"총 GPU 메모리: {total_memory:.1f} GB")
    else:
        logger.warning("CUDA를 사용할 수 없습니다. CPU로 학습합니다.")
    
    try:
        # 데이터셋 생성
        logger.info("데이터셋 로딩 중...")
        dataset = GTMatchingDataset(DATA_DIR)
        
        # 전체 데이터를 학습에 사용 (검증 데이터 없음)
        train_dataset = dataset
        
        logger.info(f"전체 데이터 학습 사용: {len(train_dataset)}개")
        
        # 데이터 로더 생성 (멀티 GPU 최적화)
        num_workers = min(8, NUM_GPUS * 2)  # GPU당 2개 워커
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        # 모델 생성 (대형 모델 사용)
        logger.info("대형 모델 초기화 중...")
        model = LargeGTMatchingModel()
        
        # 멀티 GPU 설정
        if NUM_GPUS > 1:
            logger.info(f"DataParallel로 {NUM_GPUS}개 GPU에 모델 분산")
            model = nn.DataParallel(model)
        
        # 모델 정보 출력
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        memory_gb = (total_params * 2) / (1024**3)  # float16
        
        logger.info(f"모델 파라미터: {total_params:,}개")
        logger.info(f"학습 가능 파라미터: {trainable_params:,}개")
        logger.info(f"모델 크기 (float16): {memory_gb:.2f} GB")
        best_model_loss = float('inf')
        # 학습기 생성
        trainer = GTMatchingTrainer(model, DEVICE, LEARNING_RATE)
        
        start_epoch = 0
        # # 기존 체크포인트 확인 (대형 모델이므로 새로 시작)
        # best_checkpoint_path = '/app/weights/large_gt_matching_epoch_700.pth'
        # if os.path.exists(best_checkpoint_path):
        #     logger.info("대형 모델 체크포인트 발견. 로드 중...")
        #     start_epoch = trainer.load_checkpoint(best_checkpoint_path) + 1
        # else:
        #     logger.info("새로운 대형 모델로 학습 시작")
        
        # 학습 루프
        logger.info(f"학습 시작 (에포크 {start_epoch}부터 {NUM_EPOCHS}까지)")
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            
            # 학습
            train_loss = trainer.train_epoch(train_loader, epoch)
            if train_loss < best_model_loss:
                best_model_loss = train_loss
                trainer.save_checkpoint(epoch, is_best=True)
            
            # 학습률 업데이트
            trainer.scheduler.step()
            current_lr = trainer.scheduler.get_last_lr()[0]
            trainer.learning_rates.append(current_lr)
            
            # 체크포인트 저장
            if (epoch) % SAVE_INTERVAL == 0:
                trainer.save_checkpoint(epoch)
            
        # 최종 결과 저장
        logger.info("\n학습 완료!")
        logger.info(f"전체 데이터 학습 완료 (에포크 {NUM_EPOCHS})")
        
        # 학습 곡선 저장
        trainer.plot_training_curves()
        
        # 최종 모델 저장
        trainer.save_checkpoint(NUM_EPOCHS - 1, is_best=False)
        
        logger.info("모든 작업 완료!")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main() 