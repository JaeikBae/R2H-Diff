# %%
import torch
from torchsummary import summary
from encoder import HsiEncoder, RgbSegEncoder
from unet import HsiUnet
from decoder import HsiDecoder
from diffusion_hsi import TimeEmbedding, HsiDiffusion
from torch.cuda.amp import autocast
# 모델 구성
hsi_encoder = HsiEncoder()
context_encoder = RgbSegEncoder()
net = HsiUnet()
decoder = HsiDecoder()
time_layer = TimeEmbedding(320)
model = HsiDiffusion(hsi_encoder, context_encoder, net, decoder, time_layer)
dtype = torch.float16
# 모델을 CPU에 이동
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device, dtype=dtype)

# 모델 구조 출력
print("\n=== 전체 모델 구조 ===")
summary(model, input_size=[(1, 128, 512), (1, 128, 4), (1 ,320)], device=device)
# %%