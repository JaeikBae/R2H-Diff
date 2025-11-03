# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# load all_datas_dict from npy
all_datas_dict = np.load('/app/results/202509030831/eval/all_results.npy', allow_pickle=True)
# %%
MATERIALS = [
    'blockwood', 'rubber', 'rock', 'columnwood',
    'iron', 'stainless', 'leather', 'matteglass',
    'water', 'plastic', 'plywood', 'clearglass'
]
aaaaaaaaa = {}
for i in range(len(all_datas_dict)):
    if all_datas_dict[i]['sample_name'] in MATERIALS:
        aaaaaaaaa[all_datas_dict[i]['sample_name']] = {
            'gt_matched': all_datas_dict[i]['gt_matched'],
            'hsi_gt': all_datas_dict[i]['hsi_gt'],
            'diff': np.abs(all_datas_dict[i]['gt_matched'] - all_datas_dict[i]['hsi_gt'])
        }
# %%
x , y = 200, 65
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig2, axes2 = plt.subplots(1, 1, figsize=(10, 10))

axes2.imshow(aaaaaaaaa['iron']['diff'].mean(axis=0))
for i in range(5):
    for j in range(5):
        axes2.plot(180+i*5, 55+j*5, 'ro', markersize=1)
        axes[j, i].plot(aaaaaaaaa['iron']['gt_matched'][:,55+j*5,180+i*5], 'ro', markersize=1)
        axes[j, i].plot(aaaaaaaaa['iron']['hsi_gt'][:,55+j*5,180+i*5], 'bo', markersize=1)
        axes[j, i].set_xticks([])
        axes[j, i].set_yticks([])
plt.show()
# %%
x , y = 90, 90
step = 2
fig, axes = plt.subplots(5, 5, figsize=(10, 10))
fig2, axes2 = plt.subplots(1, 1, figsize=(10, 10))

axes2.imshow(aaaaaaaaa['water']['hsi_gt'].mean(axis=0), cmap='gray')
for i in range(5):
    for j in range(5):
        axes2.plot(x+i*step, y+j*step, 'ro', markersize=1)
        axes[j, i].plot(aaaaaaaaa['water']['gt_matched'][:,y+j*step,x+i*step], 'ro', markersize=1)
        axes[j, i].plot(aaaaaaaaa['water']['hsi_gt'][:,y+j*step,x+i*step], 'bo', markersize=1)
        axes[j, i].set_xticks([])
        axes[j, i].set_yticks([])
plt.show()


# %%
# 속도 개선을 위해 numpy 벡터화 및 딕셔너리 생성 최적화

# sample_name을 바로 인덱싱할 수 있도록 딕셔너리로 변환
all_datas_dict_list = list(all_datas_dict)
sample_names = [v['sample_name'] for v in all_datas_dict_list]

# pred, gt, diff를 numpy 배열로 한 번에 추출
gt_matched_arr = np.array([v['gt_matched'] for v in all_datas_dict_list])
hsi_gt_arr = np.array([v['hsi_gt'] for v in all_datas_dict_list])

# 정규화 (벡터화)
gt_matched_min = gt_matched_arr.min(axis=(1,2,3), keepdims=True)
gt_matched_max = gt_matched_arr.max(axis=(1,2,3), keepdims=True)
gt_matched_norm = (gt_matched_arr - gt_matched_min) / (gt_matched_max - gt_matched_min + 1e-8)

hsi_gt_min = hsi_gt_arr.min(axis=(1,2,3), keepdims=True)
hsi_gt_max = hsi_gt_arr.max(axis=(1,2,3), keepdims=True)
hsi_gt_norm = (hsi_gt_arr - hsi_gt_min) / (hsi_gt_max - hsi_gt_min + 1e-8)

diff_arr = np.abs(gt_matched_arr - hsi_gt_arr)

# sample_name을 키로 하는 딕셔너리 생성 (벡터화 결과 활용)
all_datas_dict_norm = {}
for i, name in enumerate(sample_names):
    all_datas_dict_norm[name] = {
        'pred': gt_matched_norm[i],
        'gt': hsi_gt_norm[i],
        'diff': diff_arr[i]
    }

# results_matrix 생성 (딕셔너리 접근 최소화, 불필요한 반복 제거)
results_matrix = {mat1: {mat2: {} for mat2 in MATERIALS} for mat1 in MATERIALS}
for name, value in all_datas_dict_norm.items():
    if '_' in name:
        name1, name2 = name.split('_')
    else:
        name1 = name
        name2 = name
    if name1 not in MATERIALS or name2 not in MATERIALS:
        continue
    # 대칭 저장 (pred, gt, diff 모두 동일하게 저장)
    entry = {
        'pred': value['pred'],
        'gt': value['gt'],
        'diff': np.abs(value['pred'] - value['gt'])
    }
    results_matrix[name1][name2] = entry
    results_matrix[name2][name1] = entry
# %%

import matplotlib.colors as mcolors
# xticks, yticks를 활용하여 12x12 행렬 외곽에 MATERIALS 이름을 표시
DIFF_VMIN = 0.0
DIFF_VMAX = 0.1
DIFF_CMAP = 'jet'  # 낮음=파랑, 높음=빨강
norm = mcolors.Normalize(vmin=DIFF_VMIN, vmax=DIFF_VMAX)

fig, axes = plt.subplots(12, 12, figsize=(60, 60), sharex=True, sharey=True)
for i in range(12):
    for j in range(12):
        axes[i, j].imshow(
            results_matrix[MATERIALS[i]][MATERIALS[j]]['diff'].mean(axis=0),
            cmap=DIFF_CMAP, norm=norm
        )
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])
# 외곽 라벨: 아래(열 라벨), 왼쪽(행 라벨)에만 표시
for j in range(12):
    axes[-1, j].text(
        0.5, -0.08, MATERIALS[j], transform=axes[-1, j].transAxes,
        ha='center', va='top', fontsize=50, rotation=45
    )
for i in range(12):
    axes[i, 0].text(
        -0.06, 0.5, MATERIALS[i], transform=axes[i, 0].transAxes,
        ha='right', va='center', fontsize=50, rotation=0
    )

# 공통 컬러바(0.0 -> blue, 0.5 -> red)
sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(DIFF_CMAP), norm=norm)
sm.set_array([])
cbar = fig.colorbar(
    sm, ax=axes, orientation='vertical', fraction=0.025, pad=0.02, shrink=0.8, ticks=[DIFF_VMIN, DIFF_VMAX]
)
# 컬러바 폰트사이즈를 50으로 명시적으로 지정
cbar.ax.tick_params(labelsize=50)
plt.tight_layout(rect=[0.08, 0.05, 0.85, 1])  # 라벨 영역 확보
plt.show()

# %%
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
# xticks, yticks를 활용하여 12x12 행렬 외곽에 MATERIALS 이름을 표시
DIFF_VMIN = 0.0
DIFF_VMAX = 1.0
DIFF_CMAP = 'gray'  # 낮음=파랑, 높음=빨강
norm = mcolors.Normalize(vmin=DIFF_VMIN, vmax=DIFF_VMAX)

fig, axes = plt.subplots(12, 12, figsize=(60, 60), sharex=True, sharey=True)
for i in range(12):
    for j in range(12):
        if i == j: # 대각선
            axes[i, j].text(
                0.5, 0.5, MATERIALS[i], transform=axes[i, j].transAxes,
                ha='center', va='center', fontsize=50, rotation=0
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            axes[i, j].grid(False)
            # 바깥 검은색 선(스파인) 제거
            for spine in axes[i, j].spines.values():
                spine.set_visible(False)
            continue
        if i > j: # 하삼각 (PRED)
            axes[i, j].imshow(
                results_matrix[MATERIALS[i]][MATERIALS[j]]['pred'].mean(axis=0),
                cmap=DIFF_CMAP, norm=norm
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            # 테두리 및 코너 라벨
            rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=10,
                                      transform=axes[i, j].transAxes, clip_on=False, zorder=5)
            axes[i, j].add_patch(rect)
            continue
        if i < j: # 상삼각 (GT)
            axes[i, j].imshow(
                results_matrix[MATERIALS[i]][MATERIALS[j]]['gt'].mean(axis=0),
                cmap=DIFF_CMAP, norm=norm
            )
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            # 테두리 및 코너 라벨
            rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', linewidth=10,
                                      transform=axes[i, j].transAxes, clip_on=False, zorder=5)
            axes[i, j].add_patch(rect)
            continue
# 외곽 라벨: 아래(열 라벨), 왼쪽(행 라벨)에만 표시
# for j in range(12):
#     axes[-1, j].text(
#         0.5, -0.08, MATERIALS[j], transform=axes[-1, j].transAxes,
#         ha='center', va='top', fontsize=50, rotation=45
#     )
# for i in range(12):
#     axes[i, 0].text(
#         -0.06, 0.5, MATERIALS[i], transform=axes[i, 0].transAxes,
#         ha='right', va='center', fontsize=50, rotation=0
#     )
# 컬러바 폰트사이즈를 50으로 명시적으로 지정
cbar.ax.tick_params(labelsize=50)
# 범례: 상삼각(GT, 파란 테두리) / 하삼각(PRED, 빨간 테두리)
legend_handles = [
    Line2D([0], [0], color='blue', lw=20, label='GT (upper)'),
    Line2D([0], [0], color='red', lw=20, label='PRED (lower)')
]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.02), fontsize=70, frameon=False)
plt.tight_layout(rect=[0.08, 0.07, 0.85, 1])  # 하단 범례 공간 확보
plt.show()

# %%
# ============================================
# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import json
JSON_PATH = '/app/results/202509030831/eval/summary.json'
with open(JSON_PATH, 'r') as f:
    results = json.load(f)
MATERIALS = [
    'blockwood', 'rubber', 'rock', 'columnwood',
    'iron', 'stainless', 'leather', 'matteglass',
    'water', 'plastic', 'plywood', 'clearglass'
]

# %%
# Build 12x12 metric matrix from summary.json
# - Diagonal: single material metric (e.g., 'blockwood')
# - Off-diagonal: pair metric (e.g., 'blockwood_plastic' or 'plastic_blockwood')
METRIC_NAME = 'PSNR'  # options: 'GT_Matched_MSE' or any in 'GT_Matched_Metrics' such as PSNR, SSIM, RMSE, MRAE, SAM
VALUE_KEY_MSE = 'GT_Matched_MSE'
METRICS_KEY = 'GT_Matched_Metrics'
HEATMAP_CMAP = 'viridis'
ANNOTATE_VALUES = True

def getMetricValue(record, metricName):
    if metricName == VALUE_KEY_MSE:
        return record.get(VALUE_KEY_MSE)
    metrics = record.get(METRICS_KEY, {})
    return metrics.get(metricName)


def findRecordKey(materialA, materialB, dataDict):
    if materialA == materialB:
        return materialA if materialA in dataDict else None
    key1 = f"{materialA}_{materialB}"
    key2 = f"{materialB}_{materialA}"
    if key1 in dataDict:
        return key1
    if key2 in dataDict:
        return key2
    return None


values = {
    'PSNR': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float),
    'SSIM': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float),
    'RMSE': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float),
    'MRAE': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float),
    'SAM': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float)
}
for i, matI in enumerate(MATERIALS):
    for j, matJ in enumerate(MATERIALS):
        recKey = findRecordKey(matI, matJ, results)
        if recKey is None:
            continue
        record = results.get(recKey, {})
        for metricName in values.keys():
            metricVal = getMetricValue(record, metricName)
            try:
                values[metricName][i, j] = float(metricVal)
            except Exception:
                continue
# %%
VIS_ORDER = [
    'leather',
    'columnwood',
    'plywood',
    'blockwood',
    'plastic',
    'rubber',
    'water',
    'clearglass',
    'matteglass',
    'rock',
    'iron',
    'stainless'
]
VIS_ORDER_INDEX = { name: idx for idx, name in enumerate(VIS_ORDER) }
mean_values_per_material = {}
for i in range(len(MATERIALS)):
    if MATERIALS[i] not in mean_values_per_material:
        mean_values_per_material[MATERIALS[i]] = {}
    mean_values_per_material[MATERIALS[i]] = {
        'PSNR': np.nanmean(values['PSNR'][i, :]),
        'SSIM': np.nanmean(values['SSIM'][i, :]),
        'RMSE': np.nanmean(values['RMSE'][i, :]),
        'MRAE': np.nanmean(values['MRAE'][i, :]),
        'SAM': np.nanmean(values['SAM'][i, :])
    }
# pandas 없이 표 형태로 각 소재별 평균 metric 값을 출력합니다.

def printMeanMetricsTableNoPandas(meanValuesDict, order=None):
    try:
        # 지표 순서 및 헤더 정의 (요청 순서)
        metricOrder = ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']
        metricHeaders = ['MRAE\u2193', 'RMSE\u2193', 'PSNR\u2191', 'SAM\u2193', 'SSIM\u2191']
        headers = ['Material'] + metricHeaders
        # 각 행 데이터 준비
        rows = []
        if order is not None:
            orderedMaterials = [m for m in order if m in meanValuesDict]
            # order에 없는 나머지 재료들은 뒤에 사전 순으로 붙임
            remaining = sorted([m for m in meanValuesDict.keys() if m not in orderedMaterials])
            materialsSeq = orderedMaterials + remaining
        else:
            materialsSeq = list(meanValuesDict.keys())

        for material in materialsSeq:
            metrics = meanValuesDict[material]
            row = [material] + [f"{metrics[m]:.4f}" for m in metricOrder]
            rows.append(row)
        # 평균 행 계산 (선택된 materialsSeq 기준)
        if len(materialsSeq) > 0:
            avgVals = { m: float(np.nanmean([meanValuesDict[name][m] for name in materialsSeq])) for m in metricOrder }
            avgRow = ['Average'] + [f"{avgVals[m]:.4f}" for m in metricOrder]
        else:
            avgRow = ['Average', '-', '-', '-', '-', '-']
        # 컬럼별 최대 길이 계산
        colWidths = [max(len(str(row[i])) for row in ([headers] + rows + [avgRow])) for i in range(len(headers))]
        # 헤더 출력
        headerLine = " | ".join(headers[i].ljust(colWidths[i]) for i in range(len(headers)))
        print(headerLine)
        print("-" * (sum(colWidths) + 3 * (len(headers) - 1)))
        # 각 행 출력
        for row in rows:
            print(" | ".join(row[i].ljust(colWidths[i]) for i in range(len(headers))))
        # 평균 행 출력
        print("-" * (sum(colWidths) + 3 * (len(headers) - 1)))
        print(" | ".join(avgRow[i].ljust(colWidths[i]) for i in range(len(headers))))
    except Exception as e:
        print(f"테이블 출력 중 오류 발생: {e}")

printMeanMetricsTableNoPandas(mean_values_per_material, order=VIS_ORDER)
# %%
MRAE_avg = values['MRAE']
RMSE_avg = values['RMSE']
PSNR_avg = values['PSNR']
SAM_avg = values['SAM']
SSIM_avg = values['SSIM']
print(MRAE_avg.mean())
print(RMSE_avg.mean())
print(PSNR_avg.mean())
print(SAM_avg.mean())
print(SSIM_avg.mean())
# %%
print(values['MRAE'])
# %%

# 균형 잡힌 서브플롯 배치를 위한 유틸리티 (5개일 때 3+2 중앙 정렬)
def createBalancedAxes(numPanels, figsize=(12, 15)):
    try:
        import matplotlib.gridspec as gridspec
        if numPanels == 5:
            fig = plt.figure(figsize=figsize)
            gs = gridspec.GridSpec(3, 4, figure=fig)
            axes = [
                fig.add_subplot(gs[0, 0:2]),  # Row1: 2 panels
                fig.add_subplot(gs[0, 2:4]),
                fig.add_subplot(gs[1, 0:2]),  # Row2: 2 panels
                fig.add_subplot(gs[1, 2:4]),
                fig.add_subplot(gs[2, 1:3]),  # Row3: centered single panel
            ]
            return fig, axes
        cols = 3
        rows = int(np.ceil(numPanels / cols))
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten()
        for k in range(numPanels, rows * cols):
            axes[k].set_visible(False)
        return fig, list(axes[:numPanels])
    except Exception as e:
        # 예외 시 기본 플롯으로 폴백
        fig, axes = plt.subplots(1, numPanels, figsize=figsize)
        return fig, (axes if isinstance(axes, (list, tuple)) else [axes])

# %% Plot: mean metrics as bar charts (six separate subplots)
metricOrder = ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']
metricArrow = {
    'MRAE': '↓',
    'RMSE': '↓',
    'PSNR': '↑',
    'SAM': '↓',
    'SSIM': '↑'
}

# Ordered materials (same as table)
orderedMaterials = [m for m in VIS_ORDER if m in mean_values_per_material]
remainingMaterials = sorted([m for m in mean_values_per_material.keys() if m not in orderedMaterials])
materialsSeq = orderedMaterials + remainingMaterials

labels = materialsSeq + ['Average']

fig, axes = createBalancedAxes(len(metricOrder), figsize=(15, 15))

for idx, metric in enumerate(metricOrder):
    ax = axes[idx]
    vals = [mean_values_per_material[name][metric] for name in materialsSeq]
    avgVal = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
    # 정렬: 지표별 방향 적용 (↓는 오름차순, ↑는 내림차순)
    finiteIdx = [i for i, v in enumerate(vals) if np.isfinite(v)]
    nonFiniteIdx = [i for i, v in enumerate(vals) if not np.isfinite(v)]
    reverseFlag = True if metricArrow[metric] == '↑' else False
    finiteSorted = sorted(finiteIdx, key=lambda i: vals[i], reverse=reverseFlag)
    orderIdx = finiteSorted + nonFiniteIdx
    sortedMaterials = [materialsSeq[i] for i in orderIdx]
    sortedVals = [vals[i] for i in orderIdx]

    # 평균 막대 앞에 한 칸 띄우기
    labels_sorted = sortedMaterials
    labels_with_gap = labels_sorted + [''] + ['Average']
    data = np.array(sortedVals + [np.nan] + [avgVal], dtype=float)

    x = np.arange(len(labels_with_gap))
    colors = ['tab:blue'] * len(sortedVals) + ['none'] + ['tab:orange']
    # 최고/최악 표시 (Average 제외)
    dataMaterials = np.array(sortedVals, dtype=float)
    if np.any(np.isfinite(dataMaterials)):
        if metricArrow[metric] == '↓':  # 낮을수록 좋음 → 오른쪽 끝이 Best
            bestIdx = int(np.nanargmin(dataMaterials))
            worstIdx = int(np.nanargmax(dataMaterials))
        else:  # 높을수록 좋음 → 왼쪽 끝이 Best
            bestIdx = int(np.nanargmax(dataMaterials))
            worstIdx = int(np.nanargmin(dataMaterials))
        colors[bestIdx] = 'tab:green'
        colors[worstIdx] = 'tab:red'
    bars = ax.bar(x, data, color=colors)
    ax.set_title(f'{metric} {metricArrow[metric]}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_gap, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if idx == 0:
        ax.set_ylim(0,0.35)
    if idx == 1:
        ax.set_ylim(0,0.05)
    if idx == 2:
        ax.set_ylim(0,50)
        ax.set_yticks([aa for aa in range(0,50,5)])
    if idx == 3:
        ax.set_ylim(0,0.1)
    if idx == 4:
        ax.set_ylim(0,1.3)

    # Annotate values on top of bars
    for xi, bi, val in zip(x, bars, data):
        if np.isfinite(val):
            if idx != 2:
                ax.text(xi, bi.get_height(), f'{val:.4f}', ha='center', va='bottom', fontsize=10, rotation=45)
            else:
                ax.text(xi, bi.get_height(), f'{val:.2f}', ha='center', va='bottom', fontsize=10, rotation=45)
        else:
            ax.text(xi, 0, '', ha='center', va='bottom', fontsize=10, rotation=45, color='dimgray')

# 공통 범례 (Best/ Worst/ Material/ Average)
legend_handles = [
    mpatches.Patch(color='tab:green', label='Best'),
    mpatches.Patch(color='tab:red', label='Worst'),
    mpatches.Patch(color='tab:blue', label='Material'),
    mpatches.Patch(color='tab:orange', label='Average')
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=False)
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
plt.show()
# %%

total_by_metric_single = {}
# Diagonal-only single metrics (no averaging): prettier bar plots
metricOrder_single = ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']

# Use same ordered materials as above table/plots
orderedMaterials = [m for m in VIS_ORDER if m in MATERIALS]
materialsSeq = orderedMaterials

fig, axes = createBalancedAxes(len(metricOrder_single), figsize=(12, 15))

for idx, metric in enumerate(metricOrder_single):
    ax = axes[idx]
    # gather diagonal values in MATERIALS index space
    vals = []
    for m in materialsSeq:
        i = MATERIALS.index(m)
        vals.append(values[metric][i, i])
    # 정렬: 지표별 방향 적용 (↓는 오름차순, ↑는 내림차순)
    finiteIdx = [i for i, v in enumerate(vals) if np.isfinite(v)]
    nonFiniteIdx = [i for i, v in enumerate(vals) if not np.isfinite(v)]
    reverseFlag = True if metric in ['PSNR', 'SSIM'] else False
    finiteSorted = sorted(finiteIdx, key=lambda i: vals[i], reverse=reverseFlag)
    orderIdx = finiteSorted + nonFiniteIdx
    sortedMaterials = [materialsSeq[i] for i in orderIdx]
    sortedVals = [vals[i] for i in orderIdx]
    materialsSeq_sorted = sortedMaterials
    data = np.array(sortedVals, dtype=float)
    avgVal = float(np.nanmean(data)) if data.size > 0 else np.nan
    # Insert a gap (NaN) before the Average bar to match x length and colors
    data_plot = np.append(np.append(data, np.nan), avgVal)
    total_by_metric_single[metric] = data_plot
    # 평균 막대 앞에 한 칸 띄우기
    x = np.arange(len(materialsSeq_sorted) + 2)
    colors = ['tab:blue'] * len(materialsSeq_sorted) + ['none'] + ['tab:orange']
    # mark best/worst
    finiteMask = np.isfinite(data)
    if np.any(finiteMask):
        try:
            arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM'] else '↑'
            if arrow == '↓':  # 낮을수록 좋음 → 오른쪽 끝이 Best
                bestIdx = int(np.nanargmin(data))
                worstIdx = int(np.nanargmax(data))
            else:  # 높을수록 좋음 → 왼쪽 끝이 Best
                bestIdx = int(np.nanargmax(data))
                worstIdx = int(np.nanargmin(data))
            colors[bestIdx] = 'tab:green'
            colors[worstIdx] = 'tab:red'
        except Exception:
            pass

    bars = ax.bar(x, data_plot, color=colors)
    arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM'] else '↑'
    ax.set_title(f'{metric} {arrow}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(materialsSeq_sorted + [''] + ['Average'], rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for xi, bi, val in zip(x, bars, data_plot):
        if np.isfinite(val):
            ax.text(xi, bi.get_height(), f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(xi, 0, '', ha='center', va='bottom', fontsize=9, color='dimgray')

# legend
legend_handles = [
    mpatches.Patch(color='tab:green', label='Best'),
    mpatches.Patch(color='tab:red', label='Worst'),
    mpatches.Patch(color='tab:blue', label='Material'),
    mpatches.Patch(color='tab:orange', label='Average')
]
fig.legend(handles=legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=False)
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
plt.show()


# %%
total_by_metric_pair = {}   
# Pair-only mean metrics (exclude diagonal): bar charts
# Compute mean over j != i for each material i using existing 'values' and 'VIS_ORDER'
pair_mean_values_per_material = {}
for i, mat in enumerate(MATERIALS):
    pair_mean_values_per_material[mat] = {}
    for metric in ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']:
        row = values[metric][i, :].astype(float)
        if i < row.size:
            row = np.delete(row, i)
        pair_mean_values_per_material[mat][metric] = float(np.nanmean(row))

orderedMaterials = [m for m in VIS_ORDER if m in pair_mean_values_per_material]
remainingMaterials = sorted([m for m in pair_mean_values_per_material.keys() if m not in orderedMaterials])
materialsSeq = orderedMaterials + remainingMaterials
labels = materialsSeq + ['Average']

fig, axes = createBalancedAxes(5, figsize=(12, 15))

for idx, metric in enumerate(['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']):
    ax = axes[idx]
    vals = [pair_mean_values_per_material[name][metric] for name in materialsSeq]
    avgVal = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
    finiteIdx = [i for i, v in enumerate(vals) if np.isfinite(v)]
    nonFiniteIdx = [i for i, v in enumerate(vals) if not np.isfinite(v)]
    reverseFlag = True if metric in ['PSNR', 'SSIM'] else False
    finiteSorted = sorted(finiteIdx, key=lambda i: vals[i], reverse=reverseFlag)
    orderIdx = finiteSorted + nonFiniteIdx
    sortedMaterials = [materialsSeq[i] for i in orderIdx]
    sortedVals = [vals[i] for i in orderIdx]
    labels_sorted = sortedMaterials
    labels_with_gap = labels_sorted + [''] + ['Average']
    data = np.array(sortedVals + [np.nan] + [avgVal], dtype=float)
    total_by_metric_pair[metric] = data
    x = np.arange(len(labels_with_gap))
    colors = ['tab:blue'] * len(sortedVals) + ['none'] + ['tab:orange']
    dataMaterials = np.array(sortedVals, dtype=float)
    finiteMask = np.isfinite(dataMaterials)
    if np.any(finiteMask):
        arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM'] else '↑'
        if arrow == '↓':
            bestIdx = int(np.nanargmin(dataMaterials))
            worstIdx = int(np.nanargmax(dataMaterials))
        else:
            bestIdx = int(np.nanargmax(dataMaterials))
            worstIdx = int(np.nanargmin(dataMaterials))
        colors[bestIdx] = 'tab:green'
        colors[worstIdx] = 'tab:red'

    bars = ax.bar(x, data, color=colors)
    titleArrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM'] else '↑'
    ax.set_title(f'{metric} {titleArrow}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_gap, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for xi, bi, val in zip(x, bars, data):
        if np.isfinite(val):
            ax.text(xi, bi.get_height(), f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            ax.text(xi, 0, '', ha='center', va='bottom', fontsize=9, color='dimgray')

fig.legend(handles=legend_handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.02), fontsize=10, frameon=False)
plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
plt.show()

# %%
total_data = {}
for metric in metricOrder_single:
    total_data[metric] = total_by_metric_single[metric]+total_by_metric_pair[metric]
fig, axes = createBalancedAxes(5, figsize=(12, 15))

for idx, metric in enumerate(['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']):
    ax = axes[idx]
    vals = [pair_mean_values_per_material[name][metric] for name in materialsSeq]
    avgVal = float(np.nanmean(vals)) if len(vals) > 0 else np.nan
    finiteIdx = [i for i, v in enumerate(vals) if np.isfinite(v)]
    nonFiniteIdx = [i for i, v in enumerate(vals) if not np.isfinite(v)]
    reverseFlag = True if metric in ['PSNR', 'SSIM'] else False
    finiteSorted = sorted(finiteIdx, key=lambda i: vals[i], reverse=reverseFlag)
    orderIdx = finiteSorted + nonFiniteIdx
    sortedMaterials = [materialsSeq[i] for i in orderIdx]
    sortedVals = [vals[i] for i in orderIdx]
    labels_sorted = sortedMaterials
    labels_with_gap = labels_sorted + [''] + ['Average']
    data = np.array(sortedVals + [np.nan] + [avgVal], dtype=float)
    total_by_metric_pair[metric] = data
    x = np.arange(len(labels_with_gap))
    colors = ['tab:blue'] * len(sortedVals) + ['none'] + ['tab:orange']
    dataMaterials = np.array(sortedVals, dtype=float)
    finiteMask = np.isfinite(dataMaterials)
    if np.any(finiteMask):
        arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM'] else '↑'
        if arrow == '↓':
            bestIdx = int(np.nanargmin(dataMaterials))
            worstIdx = int(np.nanargmax(dataMaterials))
        else:
            bestIdx = int(np.nanargmax(dataMaterials))
            worstIdx = int(np.nanargmin(dataMaterials))
        colors[bestIdx] = 'tab:green'
        colors[worstIdx] = 'tab:red'

    bars = ax.bar(x, data, color=colors)
    titleArrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM'] else '↑'
    ax.set_title(f'{metric} {titleArrow}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_with_gap, rotation=45, ha='right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for xi, bi, val in zip(x, bars, data):
        if np.isfinite(val):
            ax.text(xi, bi.get_height(), f'{val:.4f}', ha='center', va='bottom', fontsize=9, rotation=45)
        else:
            ax.text(xi, 0, '', ha='center', va='bottom', fontsize=9, color='dimgray', rotation=45)

# %%
import matplotlib.pyplot as plt
import numpy as np

# ==========================
# 5개 지표별 12x12 히트맵 플롯
# ==========================

# 필요한 상수 및 데이터 준비
METRICS = ['MRAE', 'RMSE', 'PSNR', 'SAM', 'SSIM']
MATERIALS = [
    'blockwood', 'rubber', 'rock', 'columnwood',
    'iron', 'stainless', 'leather', 'matteglass',
    'water', 'plastic', 'plywood', 'clearglass'
]
NUM_MATERIALS = len(MATERIALS)

# 실제 데이터 로딩
import json
import os
# 각 메트릭별 최대/최소값 조사 (디버깅 및 시각화 범위 확인용)
metricMinMax = {}
for metric in METRICS:
    mat = values[metric]
    finiteVals = mat[np.isfinite(mat)]
    if finiteVals.size > 0:
        metricMinMax[metric] = {'min': float(np.min(finiteVals)), 'max': float(np.max(finiteVals))}
    else:
        metricMinMax[metric] = {'min': None, 'max': None}
print("각 메트릭별 min/max:", metricMinMax)
# %%
# 각 metric별 vmin/vmax 설정 (플롯 해상도와 일관성 위해)
VMIN_VMAX = {
    'MRAE': (0.0, 1.5),
    'RMSE': (0.0, 0.08),
    'PSNR': (20.0, 40.0),
    'SAM':  (0.0, 0.2),
    'SSIM': (0.7, 1.0)
}
CMAPS = {
    'MRAE': 'jet',
    'RMSE': 'jet',
    'PSNR': 'jet_r',
    'SAM': 'jet',
    'SSIM': 'jet_r'
}
ARROWS = {
    'MRAE': '↓',
    'RMSE': '↓',
    'PSNR': '↑',
    'SAM': '↓',
    'SSIM': '↑'
}

fig, axes = createBalancedAxes(5, figsize=(18, 20))

for idx, metric in enumerate(METRICS):
    ax = axes[idx]
    mat = values[metric]
    vmin, vmax = VMIN_VMAX[metric]
    cmap = CMAPS[metric]
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax)
    # 각 서브플롯 옆에 컬러바 추가
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # 값 텍스트 표시
    for i in range(NUM_MATERIALS):
        for j in range(NUM_MATERIALS):
            val = mat[i, j]
            if np.isfinite(val):
                if metric in ['PSNR']:
                    txt = f"{val:.1f}"
                elif metric in ['SSIM']:
                    txt = f"{val:.3f}"
                else:
                    txt = f"{val:.3f}"
                ax.text(j, i, txt, ha='center', va='center', color='black', fontsize=9)
    # 축 라벨
    ax.set_xticks(np.arange(NUM_MATERIALS))
    ax.set_yticks(np.arange(NUM_MATERIALS))
    ax.set_xticklabels(MATERIALS, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(MATERIALS, fontsize=10)
    ax.set_title(f"{metric} {ARROWS[metric]}", fontsize=18)


# %%
# 12개 소재별로 예측(pred)과 GT(gt)의 평균 스펙트럼을 한 그래프에 그리는 12개(3x4) 서브플롯 생성
fig, axes = plt.subplots(4, 3, figsize=(10, 13))
axes = axes.flatten()

# Wavelength range: 420~1728 nm
# We'll assume the number of channels equals len(pred_mean)
# and linearly space the wavelengths between 420 and 1728.
for idx, material in enumerate(MATERIALS):
    ax = axes[idx]
    # 각 소재의 diagonal (자기 자신) 샘플에서 pred, gt 추출
    try:
        pred = results_matrix[material][material]['pred']
        gt = results_matrix[material][material]['gt']
    except KeyError:
        # 데이터가 없으면 빈 그래프
        ax.set_visible(False)
        continue

    # 평균 스펙트럼 계산 (axis=1: 각 채널별 평균)
    pred_mean = pred[:,128,128]
    gt_mean = gt[:,128,128]

    # 표준편차(스펙트럼의 변동성)도 계산
    pred_std = pred.std(axis=(1, 2)) if pred.ndim == 3 else pred.std(axis=0)
    gt_std = gt.std(axis=(1, 2)) if gt.ndim == 3 else gt.std(axis=0)

    num_channels = len(pred_mean)
    wavelengths = np.linspace(420, 1728, num_channels)

    # GT(갈색), Pred(파랑) + 표준편차 영역
    ax.plot(wavelengths, gt_mean, label='GT', linewidth=3, color='orange')
    ax.plot(wavelengths, pred_mean, label='Pred', linewidth=3, linestyle='--', color='tab:blue')

    ax.set_title(material.replace('blockwood', 'wood_block')
                          .replace('columnwood', 'wood_column')
                          .replace('plywood', 'wood_plywood')
                          .replace('clearglass', 'glass_clear')
                          .replace('matteglass', 'glass_matte'), fontsize=12)
    # x축 눈금 설정: 420~1728에서 적당히 5~7개 정도로 표시
    ax.set_xlim(420, 1728)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(420, 1728, 7, dtype=int))
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

# 하단 중앙과 좌측 중앙에 라벨 추가
fig.text(0.5, 0.04, 'Wavelength (nm)', ha='center', va='center', fontsize=14)
fig.text(0.06, 0.5, 'Intensity', ha='center', va='center', rotation='vertical', fontsize=14)

# 좌하단 레전드 추가 (영문만 사용)
legend_handles = [
    plt.Line2D([0], [0], linewidth=3, color='orange', label='GT'),
    plt.Line2D([0], [0], linewidth=3, linestyle='--', color='tab:blue', label='Pred')
]
fig.legend(handles=legend_handles, loc='lower right', bbox_to_anchor=(0.98, 0.01), fontsize=12, frameon=False)

plt.tight_layout(rect=[0.1, 0.06, 0.98, 0.98])
plt.show()

# %%
# Figure 14: 소재별 채널별 에러 요약 플롯 (Median, IQR)
# - 각 소재별로, 각 채널에서의 median absolute error와 IQR(25~75%)를 시각화
# - x축: Channel, y축: Error, 각 subplot: material
# - 범례: Median, IQR (영문만 사용)
# - 한글 미사용, 경로/매직넘버 상수화, 예외처리 및 가독성 고려

NUM_MATERIALS = len(MATERIALS)
N_COLS = 3
N_ROWS = int(np.ceil(NUM_MATERIALS / N_COLS))
ERROR_LABEL = 'Error'
CHANNEL_LABEL = 'Channel'

def compute_channelwise_error(pred, gt):
    """
    예측(pred)과 정답(gt) 스펙트럼에서 채널별 median absolute error와 IQR(25~75%) 계산
    pred, gt shape: (C, H, W) 또는 (C, N)
    반환: median (C,), q25 (C,), q75 (C,)
    """
    try:
        if pred.ndim == 3:
            # (C, H, W) -> (C, H*W)
            pred_flat = pred.reshape(pred.shape[0], -1)
            gt_flat = gt.reshape(gt.shape[0], -1)
        elif pred.ndim == 2:
            pred_flat = pred
            gt_flat = gt
        else:
            raise ValueError("pred/gt shape must be (C, H, W) or (C, N)")
        abs_err = np.abs(pred_flat - gt_flat)
        median = np.median(abs_err, axis=1)
        q25 = np.percentile(abs_err, 25, axis=1)
        q75 = np.percentile(abs_err, 75, axis=1)
        return median, q25, q75
    except Exception as e:
        print(f"Error in compute_channelwise_error: {e}")
        return None, None, None

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(12, 3.5 * N_ROWS), sharex=True, sharey=True)
axes = axes.flatten()

for idx, material in enumerate(MATERIALS):
    ax = axes[idx]
    try:
        pred = results_matrix[material][material]['pred']
        gt = results_matrix[material][material]['gt']
    except KeyError:
        ax.set_visible(False)
        continue

    median, q25, q75 = compute_channelwise_error(pred, gt)
    if median is None:
        ax.set_visible(False)
        continue

    x = np.arange(len(median))
    # IQR 영역
    ax.fill_between(x, q25, q75, color='skyblue', alpha=0.5, label='IQR')
    # Median curve
    ax.plot(x, median, color='tab:blue', linewidth=2, label='Median')

    # 제목: material명 (영문, 표기 통일)
    ax.set_title(material.replace('blockwood', 'wood_block')
                          .replace('columnwood', 'wood_column')
                          .replace('plywood', 'wood_plywood')
                          .replace('clearglass', 'glass_clear')
                          .replace('matteglass', 'glass_matte'),
                 fontsize=11)
    ax.tick_params(axis='both', which='both', labelsize=9)

# 빈 subplot 숨기기
for i in range(len(MATERIALS), len(axes)):
    axes[i].set_visible(False)

# 공통 x, y 라벨
fig.text(0.5, 0.04, CHANNEL_LABEL, ha='center', va='center', fontsize=14)
fig.text(0.06, 0.5, ERROR_LABEL, ha='center', va='center', rotation='vertical', fontsize=14)

# 하단 우측 범례 (영문만)
handles = [
    plt.Line2D([0], [0], color='tab:blue', linewidth=2, label='Median'),
    plt.Line2D([0], [0], color='skyblue', linewidth=8, alpha=0.5, label='IQR')
]
fig.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.98, 0.01), fontsize=12, frameon=False)

plt.tight_layout(rect=[0.1, 0.06, 0.98, 0.98])
plt.show()

# %%
# 각 material별 예측 vs GT의 에러 맵을 subplot으로 시각화하는 코드 예시입니다.
# (이미지 예시 참고, 한글 미사용, material명 표기 통일)
import matplotlib.pyplot as plt
import numpy as np

# material명 표기 통일 함수
def get_material_label(material):
    return (material.replace('blockwood', 'wood_block')
                    .replace('columnwood', 'wood_column')
                    .replace('plywood', 'wood_plywood')
                    .replace('clearglass', 'glass_clear')
                    .replace('matteglass', 'glass_matte'))

# 하드코딩된 상수 분리
ERROR_MAP_VMIN = 0.0
ERROR_MAP_VMAX = 0.1
CMAP = 'jet'
max_error = 0.0
min_error = np.inf
def plot_error_maps(results_matrix, materials):
    global max_error, min_error
    num_materials = len(materials)
    ncols = 3
    nrows = int(np.ceil(num_materials / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.5, nrows*3.5))
    axes = axes.flatten()

    for idx, material in enumerate(materials):
        ax = axes[idx]
        try:
            pred = results_matrix[material][material]['pred']
            gt = results_matrix[material][material]['gt']
        except KeyError:
            ax.set_visible(False)
            continue

        # 예측과 GT shape 확인 및 예외 처리
        if pred.shape != gt.shape:
            ax.set_visible(False)
            print(f"Shape mismatch for {material}: pred {pred.shape}, gt {gt.shape}")
            continue

        # 에러 맵 계산 (RMSE)
        error_map = np.sqrt((pred - gt)**2)
        max_error = max(max_error, np.max(error_map))
        min_error = min(min_error, np.min(error_map))
        im = ax.imshow(error_map[128], vmin=ERROR_MAP_VMIN, vmax=ERROR_MAP_VMAX, cmap=CMAP)
        ax.set_title(get_material_label(material), fontsize=10)
        ax.axis('off')

    # 빈 subplot 숨기기
    for i in range(num_materials, len(axes)):
        axes[i].set_visible(False)

    # colorbar 추가 (오른쪽 중앙)
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.025, pad=0.04, ticks=[0.0, 0.1])
    plt.tight_layout(rect=[0.08, 0.05, 0.85, 1])
    plt.show()
plot_error_maps(results_matrix, MATERIALS)
print(f"max_error: {max_error}, min_error: {min_error}")

# %%

# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
from tensorly.decomposition import tucker
import tensorly as tl
import PIL.Image as Image
import os
tl.set_backend('pytorch')
A = [2,4,8,16,32,64,128,256,512]
B = [1,2,4,8,16,32,64,128,256]
C = [1,2,4,8,16,32,64,128,256]
hsi_tensor = torch.randn(512, 256, 256).to(device='cuda')
hsi_tensor = hsi_tensor.float().to(device='cuda')
print(f"hsi_tensor.shape: {hsi_tensor.shape}")
results = []
orig_size = 512 * 256 * 256
target_rank = (64, 128, 128)
target_log_abc = np.log2(target_rank[0]*target_rank[1]*target_rank[2])
target_mse = None
target_size_reduced = None
for ia, a in enumerate(A):
    for ib, b in enumerate(B):
        for ic, c in enumerate(C):
            try:
                core, factors = tucker(hsi_tensor, rank=(a, b, c))
                re = tl.tucker_to_tensor((core, factors))
                rmse = torch.sqrt(torch.nn.functional.mse_loss(re, hsi_tensor)).item()
                tucker_size = a * b * c + a * 512 + b * 256 + c * 256
                size_reduced = (1 - (tucker_size / orig_size)) * 100
                log_abc = np.log2(a*b*c)
                results.append({'log_abc': log_abc,
                                'abc': a*b*c,
                                'a': a, 'b': b, 'c': c,
                                'rmse': rmse,
                                'size_reduced': size_reduced})
                print(f"a: {a}, b: {b}, c: {c}, rmse: {rmse}, size_reduced: {size_reduced}")
                if log_abc == target_log_abc:
                    target_rmse = rmse
                    target_size_reduced = size_reduced
            except Exception as e:
                print(f"Error in tucker: {e}")
                continue

# numpy array로 변환, x축은 log_abc
results_arr = np.array([
    (item['log_abc'], item['rmse'], item['size_reduced'], item['a'], item['b'], item['c']) for item in results
])
log_abc = results_arr[:, 0]
rmse = results_arr[:, 1]
size_reduced = results_arr[:, 2]
a_list, b_list, c_list = results_arr[:, 3], results_arr[:, 4], results_arr[:, 5]
print(f"target_rmse: {target_rmse}, target_log_abc: {target_log_abc}")
# %%
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
# fig.suptitle('Tucker Decomposition Performance: Compression vs Reconstruction Error', fontsize=16, fontweight='bold')

# 왼쪽: MSE vs 2^x
ax = axes[0]
sc0 = ax.scatter(log_abc, rmse)
ax.set_xlabel('log₂(R₁ × R₂ × R₃) - Compression Level (core size, logarithmic scale)', fontsize=12)
ax.set_ylabel('Reconstruction Error (RMSE)', fontsize=12)
ax.set_title('Reconstruction Error by Compression Level', fontsize=13)
ax.set_xticks(np.unique(log_abc))
ax.set_xticklabels([int(x) for x in np.unique(log_abc)])
ax.grid(True, axis='both', linestyle='--', alpha=0.7)
ax.scatter(target_log_abc, target_rmse, color='red', marker='*', s=200)
ax.legend()

# 오른쪽: Size reduction vs 2^x
ax = axes[1]
sc1 = ax.scatter(log_abc, size_reduced)
ax.set_xlabel('log₂(R₁ × R₂ × R₃) - Compression Level (core size, logarithmic scale)', fontsize=12)
ax.set_ylabel('Size Reduction (%)', fontsize=12)
ax.set_title('Compression Efficiency by Compression Level', fontsize=13)
ax.set_xticks(np.unique(log_abc))
ax.set_xticklabels([int(x) for x in np.unique(log_abc)])
ax.grid(True, axis='both', linestyle='--', alpha=0.7)
ax.scatter(target_log_abc, target_size_reduced, color='red', marker='*', s=200)
ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# %%
by_abc = []
by_abc_size = []
aa = np.unique(log_abc)
aa.sort()
for abc in aa:
    at_abc = rmse[log_abc == abc]
    at_abc_size = size_reduced[log_abc == abc]
    by_abc.append(at_abc.mean())
    by_abc_size.append(at_abc_size.mean())
by_abc = np.array(by_abc)
by_abc_size = np.array(by_abc_size)
diff = by_abc[1:] - by_abc[:-1]
diff_size = by_abc_size[1:] - by_abc_size[:-1]
for a in range(len(aa)):
    print(aa[a], by_abc[a], by_abc_size[a], diff[a], diff_size[a])
