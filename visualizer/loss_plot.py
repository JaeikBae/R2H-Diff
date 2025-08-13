# %%
# read the log file and plot the loss curve
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# %matplotlib inline
save_dir = '/app/visualizer/gen_images'
loss_log_base = '/app/weights/202508030846'

losses = []
with open(loss_log_base + '/losses.txt', 'r') as f:
    for line in f:
        try: # ignore config lines
            losses.append(float(line.strip()))
        except:
            pass
n = 300
losses = losses[n:]
# losses = [np.mean(losses[i:i+10]) for i in range(0, len(losses), 10)]
print(max(losses), min(losses))
# scale the loss values
losses = np.array(losses)
print(f'Number of iterations: {len(losses)}')

# Plotly로 변환
fig = go.Figure()
fig.add_trace(go.Scatter(x=np.arange(len(losses)), y=losses, mode='lines', name='Loss'))
# trend line
z = np.polyfit(np.arange(len(losses)), losses, 1)
p = np.poly1d(z)
fig.add_trace(go.Scatter(x=np.arange(len(losses)), y=p(np.arange(len(losses))), 
                        mode='lines', line=dict(dash='dash', color='red'), name='Trend'))
print(f'Trend line: {z[0]:.8f}x + {z[1]:.4f}')
fig.update_layout(title='Loss curve', xaxis_title='Iteration', yaxis_title='Loss')
fig.show()

# %% plot_losses.py
import os
import ast
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# losses.txt 경로
LOSS_FILE = loss_log_base + '/epoch_losses.txt'

# 읽어온 데이터를 담을 리스트
epochs, main_l, recon_l, f_l = [], [], [], []

with open(LOSS_FILE, 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        epoch, main, recon, f = parts
        epochs.append(int(epoch))
        main_l.append(float(main))
        recon_l.append(float(recon))
        f_l.append(float(f))
s = n

epochs = epochs[s:]
main_l = main_l[s:]
recon_l = recon_l[s:]
f_l = f_l[s:]

# 서브플롯으로 각각의 loss를 표시
# 윗줄: recon loss 하나만, 아래줄: main과 f loss를 반반으로 2개 플롯
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=['Recon Loss', '', 'Main Loss', 'F Loss'],
    specs=[[{"secondary_y": False, "colspan": 2}, None],
           [{"secondary_y": False}, {"secondary_y": False}]],
    row_heights=[0.7, 0.3]  # recon loss를 더 길게 연장
)

# recon loss를 첫 번째 서브플롯에 추가 (전체 너비 사용)
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=recon_l, 
        mode='lines', 
        name='Recon Loss',
        line=dict(color='blue'),
        showlegend=True
    ),
    row=1, col=1
)

# recon loss 트렌드 라인 추가
z_recon = np.polyfit(epochs, recon_l, 1)
p_recon = np.poly1d(z_recon)
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=p_recon(epochs), 
        mode='lines', 
        name='Recon Trend',
        line=dict(dash='dash', color='blue'),
        showlegend=False
    ),
    row=1, col=1
)
print(f'Recon Loss Trend: {z_recon[0]:.8f}x + {z_recon[1]:.4f}')

# main loss를 두 번째 줄 왼쪽에 추가
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=main_l, 
        mode='lines', 
        name='Main Loss',
        line=dict(color='red'),
        showlegend=True
    ),
    row=2, col=1
)

# main loss 트렌드 라인 추가
z_main = np.polyfit(epochs, main_l, 1)
p_main = np.poly1d(z_main)
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=p_main(epochs), 
        mode='lines', 
        name='Main Trend',
        line=dict(dash='dash', color='red'),
        showlegend=False
    ),
    row=2, col=1
)
print(f'Main Loss Trend: {z_main[0]:.8f}x + {z_main[1]:.4f}')

# f loss를 두 번째 줄 오른쪽에 추가
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=f_l, 
        mode='lines', 
        name='F Loss',
        line=dict(color='green'),
        showlegend=True
    ),
    row=2, col=2
)

# f loss 트렌드 라인 추가
z_f = np.polyfit(epochs, f_l, 1)
p_f = np.poly1d(z_f)
fig.add_trace(
    go.Scatter(
        x=epochs, 
        y=p_f(epochs), 
        mode='lines', 
        name='F Trend',
        line=dict(dash='dash', color='green'),
        showlegend=False
    ),
    row=2, col=2
)
print(f'F Loss Trend: {z_f[0]:.8f}x + {z_f[1]:.4f}')

fig.update_layout(
    title='Training Loss Curves - Recon Loss (Top) + Main/F Loss (Bottom Split)',
    height=600,
    width=1200,
    showlegend=True
)

# 각 서브플롯의 축 레이블 설정
fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_yaxes(title_text="Loss", row=1, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=1)
fig.update_yaxes(title_text="Loss", row=2, col=1)
fig.update_xaxes(title_text="Epoch", row=2, col=2)
fig.update_yaxes(title_text="Loss", row=2, col=2)

# 첫 번째 줄의 축 레이블 설정 (recon loss가 전체 너비 사용)
fig.update_xaxes(title_text="Epoch", row=1, col=1)
fig.update_yaxes(title_text="Loss", row=1, col=1)

fig.show()

# %%
import numpy as np
import plotly.graph_objects as go

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        s = 0.012

        steps = np.arange(0, num_diffusion_timesteps + 1, dtype=np.float64)
        t_ratio = steps / num_diffusion_timesteps

        alphas_cumprod = np.cos(((t_ratio + s) / (1 + s)) * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

        betas = np.zeros(num_diffusion_timesteps, dtype=np.float64)
        for i in range(num_diffusion_timesteps): 
            alpha_bar_t = alphas_cumprod[i + 1]
            alpha_bar_prev = alphas_cumprod[i]
            beta_t = 1.0 - (alpha_bar_t / alpha_bar_prev)
            betas[i] = np.clip(beta_t, a_min=0.0, a_max=0.999)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

# 예시 파라미터
T = 500
beta_start, beta_end = 5e-6, 1.5e-2
schedules = ["quad", "linear", "const", "jsd", "sigmoid", "cosine"]

# Plotly로 변환
fig = go.Figure()
colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
for i, name in enumerate(schedules):
    betas = get_beta_schedule(name, beta_start=beta_start, beta_end=beta_end, num_diffusion_timesteps=T)
    fig.add_trace(go.Scatter(
        x=np.arange(1, T+1), 
        y=(1-betas).cumprod(0), 
        mode='lines', 
        name=name,
        line=dict(color=colors[i])
    ))

fig.update_layout(
    title="Alpha Schedule",
    xaxis_title="Diffusion timestep",
    yaxis_title="Alpha",
    showlegend=True
)
fig.show()
# %%
