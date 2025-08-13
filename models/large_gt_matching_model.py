#!/usr/bin/env python3
"""
대형 GTMatchingModel
더 깊고 넓은 구조로 설계된 강력한 모델
train/eval 모드에서 일관된 동작 (InstanceNorm2d 사용)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeResidualBlock(nn.Module):
    """대형 Residual Block - 더 많은 채널과 깊은 구조"""
    def __init__(self, channels, expansion=2):
        super(LargeResidualBlock, self).__init__()
        expanded_channels = channels * expansion
        
        self.conv1 = nn.Conv2d(channels, expanded_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(expanded_channels)  # LayerNorm -> InstanceNorm2d
        self.conv2 = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(expanded_channels)  # LayerNorm -> InstanceNorm2d
        self.conv3 = nn.Conv2d(expanded_channels, channels, kernel_size=3, padding=1)
        self.in3 = nn.InstanceNorm2d(channels)  # LayerNorm -> InstanceNorm2d
        
        # Skip connection을 위한 1x1 conv (채널 수가 다른 경우)
        self.shortcut = nn.Sequential()
        if expansion != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1),
                nn.InstanceNorm2d(channels)  # LayerNorm -> InstanceNorm2d
            )
        
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.in1(self.conv1(x)))
        out = F.relu(self.in2(self.conv2(out)))
        out = self.in3(self.conv3(out))
        
        out += residual
        out = F.relu(out)
        return out


class AttentionBlock(nn.Module):
    """Channel Attention Block"""
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class LargeGTMatchingModel(nn.Module):
    """
    대형 Ground Truth Matching Model
    더 깊고 넓은 구조로 설계
    train/eval 모드에서 일관된 동작 (InstanceNorm2d 사용)
    """
    
    def __init__(self, input_channels=512, output_channels=512):
        super(LargeGTMatchingModel, self).__init__()
        
        # 1. 입력 처리: 512 -> 512 (더 많은 정보 보존)
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),  # LayerNorm -> InstanceNorm2d
            nn.ReLU(inplace=True),
            AttentionBlock(512)
        )
        
        # 2. 첫 번째 블록 그룹: 512 -> 512
        self.block_group1 = nn.Sequential(
            LargeResidualBlock(512, expansion=2),
            LargeResidualBlock(512, expansion=2),
            LargeResidualBlock(512, expansion=2),
            AttentionBlock(512)
        )
        
        # 3. 두 번째 블록 그룹: 512 -> 512
        self.block_group2 = nn.Sequential(
            LargeResidualBlock(512, expansion=2),
            LargeResidualBlock(512, expansion=2),
            LargeResidualBlock(512, expansion=2),
            AttentionBlock(512)
        )
        
        # 4. 세 번째 블록 그룹: 512 -> 512
        self.block_group3 = nn.Sequential(
            LargeResidualBlock(512, expansion=2),
            LargeResidualBlock(512, expansion=2),
            LargeResidualBlock(512, expansion=2),
            AttentionBlock(512)
        )
        
        # 5. 출력 처리: 512 -> 512
        self.output_conv = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.InstanceNorm2d(512),  # LayerNorm -> InstanceNorm2d
            nn.ReLU(inplace=True),
            AttentionBlock(512)
        )
        
        # 6. 최종 출력: 입력 + 보정값 (Residual Learning)
        self.final_conv = nn.Conv2d(512, output_channels, kernel_size=3, padding=1)
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, 512, H, W) - generated HSI data
        Returns:
            Output tensor (B, 512, H, W) - corrected HSI data
        """
        # 입력 처리
        out = self.input_conv(x)
        
        # 중간 처리 (3개 블록 그룹)
        out = self.block_group1(out)
        out = self.block_group2(out)
        out = self.block_group3(out)
        
        # 출력 처리
        out = self.output_conv(out)
        
        # 최종 출력: 입력 + 보정값 (Residual Learning)
        correction = self.final_conv(out)
        # 보정값을 작게 스케일링 (0.01 배율)
        correction = correction * 0.01
        output = x + correction  # Residual connection
        
        return output


def test_large_model():
    """대형 모델 테스트 - train/eval 모드 일관성 확인 (InstanceNorm2d 사용)"""
    try:
        # 모델 생성
        model = LargeGTMatchingModel()
        
        # 테스트 입력
        batch_size = 2
        height, width = 256, 256
        input_tensor = torch.randn(batch_size, 512, height, width, dtype=torch.float32)
        
        print("=== 대형 GTMatchingModel 테스트 (InstanceNorm2d 사용) ===")
        print(f"입력 형태: {input_tensor.shape}")
        
        # Train 모드에서 테스트
        model.train()
        with torch.no_grad():
            train_output = model(input_tensor)
        
        # Eval 모드에서 테스트
        model.eval()
        with torch.no_grad():
            eval_output = model(input_tensor)
        
        # Train/Eval 모드 결과 비교
        output_diff = torch.abs(train_output - eval_output).max().item()
        
        print(f"Train 모드 출력 형태: {train_output.shape}")
        print(f"Eval 모드 출력 형태: {eval_output.shape}")
        print(f"Train/Eval 출력 차이 (최대): {output_diff:.8f}")
        
        if output_diff < 1e-6:
            print("✅ Train/Eval 모드 일관성 확인됨!")
        else:
            print("⚠️ Train/Eval 모드에서 차이 발견")
        
        print(f"입력 범위: [{input_tensor.min():.4f}, {input_tensor.max():.4f}]")
        print(f"출력 범위: [{train_output.min():.4f}, {train_output.max():.4f}]")
        
        # 파라미터 수 계산
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"총 파라미터: {total_params:,}")
        print(f"학습 가능 파라미터: {trainable_params:,}")
        
        # 메모리 사용량
        memory_gb = (total_params * 4) / (1024**3)
        print(f"모델 크기: {memory_gb:.2f} GB")
        
        # Residual Learning 확인
        correction = train_output - input_tensor
        print(f"보정값 범위: [{correction.min():.4f}, {correction.max():.4f}]")
        print(f"보정값 평균: {correction.mean():.6f}")
        print(f"보정값 표준편차: {correction.std():.6f}")
        
        print("✅ 대형 모델 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 대형 모델 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_large_model() 