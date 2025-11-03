#!/usr/bin/env python3
"""
대형 UpsampleModel
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


class UpsampleModel(nn.Module):
    """
    대형 Upsample Model
    더 깊고 넓은 구조로 설계
    train/eval 모드에서 일관된 동작 (InstanceNorm2d 사용)
    """
    
    def __init__(self, input_channels=512, output_channels=512):
        super(UpsampleModel, self).__init__()
        
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