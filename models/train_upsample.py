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
from upsample_model import UpsampleModel
from torch.cuda.amp import GradScaler, autocast

TIME = time.strftime("%Y%m%d%H%M")
DATA_DIR = "/app/align_data2"
WEIGHTS_DIR = f"/app/weights/residual_{TIME}"
RESULTS_DIR = f"/app/results/residual_{TIME}"
VISUALIZATION_DIR = f"/app/results/visualizations/residual_{TIME}"

os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10000
SAVE_INTERVAL = 50
VISUALIZATION_INTERVAL = 10
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0

# Seed
def setSeed(seed):
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
        print(f"Seed error: {e}")

setSeed(SEED)


# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{RESULTS_DIR}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class UpsampleDataset(Dataset):
    
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        self.data_files = []
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        for file_path in npy_files:
            if not file_path.endswith("_gt.npy"):
                gt_file = file_path.replace(".npy", "_gt.npy")
                if os.path.exists(gt_file):
                    self.data_files.append({
                        'data': file_path,
                        'gt': gt_file
                    })
        
        logger.info(f"Dataset loaded: {len(self.data_files)} files")
        
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx):
        data_path = self.data_files[idx]['data']
        gt_path = self.data_files[idx]['gt']
        
        data = np.load(data_path).astype(np.float16)
        gt_data = np.load(gt_path).astype(np.float16)
        if len(data.shape) == 3:
            data = torch.from_numpy(data).float()
            gt_data = torch.from_numpy(gt_data).float()
        else:
            raise ValueError(f"Unexpected data shape: {data.shape}")
        
        if self.transform:
            data = self.transform(data)
            gt_data = self.transform(gt_data)
        return data_path, data, gt_data


class VisualizationHelper:
    
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def visualize_comparison(self, input_data, output_data, gt_data, epoch, batch_idx=0, sample_idx=0):
        """
        Compare input/output/ground truth (5 channels, 5x3)
        """
        try:
            input_np = input_data[batch_idx].cpu().numpy()
            output_np = output_data[batch_idx].cpu().numpy()
            gt_np = gt_data[batch_idx].cpu().numpy()
            
            channels = [0, 127, 255, 383, 511]
            
            fig, axes = plt.subplots(5, 3, figsize=(12, 20))
            
            for i, ch in enumerate(channels):
                axes[i, 0].imshow(input_np[ch], cmap='gray', vmin=0, vmax=1)
                axes[i, 0].set_title(f'Input Ch{ch} (Epoch {epoch})')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(output_np[ch], cmap='gray', vmin=0, vmax=1)
                axes[i, 1].set_title(f'Output Ch{ch} (Epoch {epoch})')
                axes[i, 1].axis('off')
                
                axes[i, 2].imshow(gt_np[ch], cmap='gray', vmin=0, vmax=1)
                axes[i, 2].set_title(f'GT Ch{ch} (Epoch {epoch})')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            
            filename = f'comparison_epoch_{epoch:03d}_batch_{batch_idx:02d}_sample_{sample_idx:02d}.png'
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved: {filename}")
            
        except Exception as e:
            logger.error(f"Visualization error: {e}")
    
    def visualize_spectrum_comparison(self, input_data, output_data, gt_data, epoch, batch_idx=0, sample_idx=0):
        """
        Compare spectrum (specific pixel)
        """
        try:
            input_np = input_data[batch_idx, :, sample_idx, sample_idx].cpu().numpy()
            output_np = output_data[batch_idx, :, sample_idx, sample_idx].cpu().numpy()
            gt_np = gt_data[batch_idx, :, sample_idx, sample_idx].cpu().numpy()
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            wavelengths = np.arange(512)
            
            axes[0].plot(wavelengths, input_np, 'b-', alpha=0.7, label='Input')
            axes[0].plot(wavelengths, output_np, 'r-', alpha=0.7, label='Output')
            axes[0].plot(wavelengths, gt_np, 'g-', alpha=0.7, label='Ground Truth')
            axes[0].set_xlabel('Channel')
            axes[0].set_ylabel('Intensity')
            axes[0].set_title(f'Spectrum Comparison (Epoch {epoch})')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
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
            
            filename = f'spectrum_epoch_{epoch:03d}_batch_{batch_idx:02d}_sample_{sample_idx:02d}.png'
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Spectrum visualization saved: {filename}")
            
        except Exception as e:
            logger.error(f"Spectrum visualization error: {e}")


class UpsampleTrainer:
    
    def __init__(self, model, device, learning_rate=5e-5):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=1e-5,
            eps=1e-8
        )
        
        # Cosine Annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=NUM_EPOCHS,
            eta_min=1e-7
        )
        
        # Loss function (Huber Loss - more stable)
        self.criterion = nn.HuberLoss(delta=0.1)
        
        # GradScaler for Mixed Precision Training
        self.scaler = GradScaler()
        
        # Training history
        self.train_losses = []
        self.learning_rates = []
        
        # Track best model (full dataset training, so use train loss)
        self.best_train_loss = float('inf')
        self.best_epoch = 0

        self.visualizer = VisualizationHelper(VISUALIZATION_DIR)
        
        logger.info(f"Trainer initialized (device: {device})")
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        total_input_gt_mse = 0.0
        total_output_gt_mse = 0.0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"{epoch}/{NUM_EPOCHS}")
        
        for batch_idx, (data_path, data, gt_data) in enumerate(progress_bar):
            try:
                data = data.to(self.device, non_blocking=True)
                gt_data = gt_data.to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, gt_data)
                    
                    input_gt_mse = torch.mean((data - gt_data) ** 2).item()
                    output_gt_mse = torch.mean((output - gt_data) ** 2).item()
                
                self.scaler.scale(loss).backward()
                
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                
                total_loss += loss.item()
                total_input_gt_mse += input_gt_mse
                total_output_gt_mse += output_gt_mse
                
                if (epoch + 1) % VISUALIZATION_INTERVAL == 0 and batch_idx == 0:
                    with torch.no_grad():
                        self.visualizer.visualize_comparison(
                            data, output, gt_data, epoch + 1, batch_idx, 0
                        )
                        self.visualizer.visualize_spectrum_comparison(
                            data, output, gt_data, epoch + 1, batch_idx, 0
                        )
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'Avg': f'{total_loss/(batch_idx+1):.6f}',
                    'Input-GT MSE': f'{input_gt_mse:.6f}',
                    'Output-GT MSE': f'{output_gt_mse:.6f}',
                    'Improvement': f'{((input_gt_mse - output_gt_mse) / input_gt_mse * 100):.2f}%'
                })
                
            except Exception as e:
                logger.error(f"Batch training error (batch {batch_idx}): {e}")
                continue
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        
        self.train_losses.append(avg_loss)
        
        if avg_loss < self.best_train_loss:
            self.best_train_loss = avg_loss
            self.best_epoch = epoch
        
        return avg_loss
    

    
    def save_checkpoint(self, epoch, is_best=False):
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
            
            if is_best:
                best_path = os.path.join(WEIGHTS_DIR, 'upsample_best.pth')
                torch.save(checkpoint, best_path)
                logger.info(f"Best model saved: {best_path}")
            else:
                checkpoint_path = os.path.join(WEIGHTS_DIR, f'upsample_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
            
        except Exception as e:
            logger.error(f"Checkpoint save error: {e}")
    
    def load_checkpoint(self, checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            state_dict = checkpoint['model_state_dict']
            
            if isinstance(self.model, nn.DataParallel):
                # Add 'module.' prefix if not present
                if not any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        new_key = f'module.{key}'
                        new_state_dict[new_key] = value
                    state_dict = new_state_dict
            else:
                # Remove 'module.' prefix if present
                if any(key.startswith('module.') for key in state_dict.keys()):
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('module.'):
                            new_key = key[7:]
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
            
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint['epoch']
            
        except Exception as e:
            logger.error(f"Checkpoint load error: {e}")
            return 0
    
    def plot_training_curves(self):
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
            
            logger.info("Training curves saved")
            
        except Exception as e:
            logger.error(f"Training curves save error: {e}")


def main():
    logger.info("Upsample training started")
    
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    if torch.cuda.is_available():
        logger.info(f"Multi GPU setup: {NUM_GPUS} GPUs")
        for i in range(NUM_GPUS):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        total_memory = sum(torch.cuda.get_device_properties(i).total_memory for i in range(NUM_GPUS)) / (1024**3)
        logger.info(f"Total GPU memory: {total_memory:.1f} GB")
    else:
        logger.warning("CUDA not available. Training on CPU.")
    
    try:
        logger.info("Loading dataset...")
        dataset = UpsampleDataset(DATA_DIR)
        
        train_dataset = dataset
        
        logger.info(f"Using all data: {len(train_dataset)} samples")
        
        num_workers = min(8, NUM_GPUS * 2)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        logger.info("Initializing large model...")
        model = UpsampleModel()
        
        if NUM_GPUS > 1:
            logger.info(f"DataParallel on {NUM_GPUS} GPUs")
            model = nn.DataParallel(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        memory_gb = (total_params * 2) / (1024**3)  # float16
        
        logger.info(f"Model parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Model size (float16): {memory_gb:.2f} GB")
        best_model_loss = float('inf')
        trainer = UpsampleTrainer(model, DEVICE, LEARNING_RATE)
        
        start_epoch = 0
        best_checkpoint_path = '/app/weights/align_202508071057/upsample_best.pth'
        if os.path.exists(best_checkpoint_path):
            logger.info("Large model checkpoint found. Loading...")
            start_epoch = trainer.load_checkpoint(best_checkpoint_path) + 1
        else:
            logger.info("Starting new large model training")
        
        logger.info(f"Training started (epoch {start_epoch} to {NUM_EPOCHS})")
        
        for epoch in range(start_epoch, NUM_EPOCHS):
            
            train_loss = trainer.train_epoch(train_loader, epoch)
            if train_loss < best_model_loss:
                best_model_loss = train_loss
                trainer.save_checkpoint(epoch, is_best=True)
            
            trainer.scheduler.step()
            current_lr = trainer.scheduler.get_last_lr()[0]
            trainer.learning_rates.append(current_lr)
            
            if (epoch) % SAVE_INTERVAL == 0:
                trainer.save_checkpoint(epoch)
            
        logger.info("\nTraining completed!")
        logger.info(f"Full dataset training completed (epoch {NUM_EPOCHS})")
        
        trainer.plot_training_curves()
        
        trainer.save_checkpoint(NUM_EPOCHS - 1, is_best=False)
        
        logger.info("All tasks completed!")
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise


if __name__ == "__main__":
    main() 