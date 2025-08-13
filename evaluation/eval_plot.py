# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# load all_datas_dict from npy
all_datas_dict = np.load('/app/models/all_datas_dict.npy', allow_pickle=True).item()

all_datas_dict_norm = {}
for key, value in all_datas_dict.items():
    all_datas_dict_norm[key] = {
        'pred': (value['pred'] - value['pred'].min()) / (value['pred'].max() - value['pred'].min()),
        'gt': (value['gt'] - value['gt'].min()) / (value['gt'].max() - value['gt'].min()),
        'diff': np.abs(value['pred'] - value['gt'])
    }
# %%
MATERIALS = [
    'blockwood', 'rubber', 'rock', 'columnwood',
    'iron', 'stainless', 'leather', 'matteglass',
    'water', 'plastic', 'plywood', 'clearglass'
]
results_matrix = {}
i = 5
for idx, (key, value) in enumerate(all_datas_dict_norm.items()):
    if '_' in key:
        name1, name2 = key.split('_')
    else :
        name1 = key
        name2 = key
    if name1 not in MATERIALS or name2 not in MATERIALS:
        continue
    if name1 not in results_matrix:
        results_matrix[name1] = {}
    if name2 not in results_matrix[name1]:
        results_matrix[name1][name2] = {}
    if name2 not in results_matrix:
        results_matrix[name2] = {}
    if name1 not in results_matrix[name2]:
        results_matrix[name2][name1] = {}
    results_matrix[name1][name2] = {
        'pred': value['pred'],
        'gt': value['gt'],
        'diff': np.abs(value['pred'] - value['gt'])
    }
    results_matrix[name2][name1] = {
        'pred': value['pred'],
        'gt': value['gt'],
        'diff': np.abs(value['pred'] - value['gt'])
    }
    
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
            rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='red', linewidth=4,
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
            rect = mpatches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='blue', linewidth=4,
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
JSON_PATH = '/app/results/align_202508071057_202508070233_epoch_2000/eval/summary.json'
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
METRIC_NAME = 'PSNR'  # options: 'GT_Matched_MSE' or any in 'GT_Matched_Metrics' such as PSNR, SSIM, RMSE, MRAE, SAM, FID
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
    'SAM': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float),
    'FID': np.full((len(MATERIALS), len(MATERIALS)), np.nan, dtype=float)
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
        'SAM': np.nanmean(values['SAM'][i, :]),
        'FID': np.nanmean(values['FID'][i, :])
    }
# pandas 없이 표 형태로 각 소재별 평균 metric 값을 출력합니다.

def printMeanMetricsTableNoPandas(meanValuesDict, order=None):
    try:
        # 지표 순서 및 헤더 정의 (요청 순서)
        metricOrder = ['MRAE', 'RMSE', 'SAM', 'FID', 'PSNR', 'SSIM']
        metricHeaders = ['MRAE\u2193', 'RMSE\u2193', 'SAM\u2193', 'FID\u2193', 'PSNR\u2191', 'SSIM\u2191']
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
            row = [material] + [f"{metrics[m]:.2f}" for m in metricOrder]
            rows.append(row)
        # 평균 행 계산 (선택된 materialsSeq 기준)
        if len(materialsSeq) > 0:
            avgVals = { m: float(np.nanmean([meanValuesDict[name][m] for name in materialsSeq])) for m in metricOrder }
            avgRow = ['Average'] + [f"{avgVals[m]:.2f}" for m in metricOrder]
        else:
            avgRow = ['Average', '-', '-', '-', '-', '-', '-']
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

# %% Plot: mean metrics as bar charts (six separate subplots)
metricOrder = ['MRAE', 'RMSE', 'SAM', 'FID', 'PSNR', 'SSIM']
metricArrow = {
    'MRAE': '↓',
    'RMSE': '↓',
    'SAM': '↓',
    'FID': '↓',
    'PSNR': '↑',
    'SSIM': '↑'
}

# Ordered materials (same as table)
orderedMaterials = [m for m in VIS_ORDER if m in mean_values_per_material]
remainingMaterials = sorted([m for m in mean_values_per_material.keys() if m not in orderedMaterials])
materialsSeq = orderedMaterials + remainingMaterials

labels = materialsSeq + ['Average']

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes = axes.flatten()

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

    # Annotate values on top of bars
    for xi, bi, val in zip(x, bars, data):
        if np.isfinite(val):
            ax.text(xi, bi.get_height(), f'{val:.2f}', ha='center', va='bottom', fontsize=9, rotation=0)
        else:
            ax.text(xi, 0, '', ha='center', va='bottom', fontsize=9, rotation=0, color='dimgray')

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
# Diagonal-only single metrics (no averaging): prettier bar plots
metricOrder_single = ['MRAE', 'RMSE', 'SAM', 'FID', 'PSNR', 'SSIM']

# Use same ordered materials as above table/plots
orderedMaterials = [m for m in VIS_ORDER if m in MATERIALS]
materialsSeq = orderedMaterials

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes = axes.flatten()

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

    # 평균 막대 앞에 한 칸 띄우기
    x = np.arange(len(materialsSeq_sorted) + 2)
    colors = ['tab:blue'] * len(materialsSeq_sorted) + ['none'] + ['tab:orange']
    # mark best/worst
    finiteMask = np.isfinite(data)
    if np.any(finiteMask):
        try:
            arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM', 'FID'] else '↑'
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
    arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM', 'FID'] else '↑'
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
# Pair-only mean metrics (exclude diagonal): bar charts
# Compute mean over j != i for each material i using existing 'values' and 'VIS_ORDER'
pair_mean_values_per_material = {}
for i, mat in enumerate(MATERIALS):
    pair_mean_values_per_material[mat] = {}
    for metric in ['MRAE', 'RMSE', 'SAM', 'FID', 'PSNR', 'SSIM']:
        row = values[metric][i, :].astype(float)
        if i < row.size:
            row = np.delete(row, i)
        pair_mean_values_per_material[mat][metric] = float(np.nanmean(row))

orderedMaterials = [m for m in VIS_ORDER if m in pair_mean_values_per_material]
remainingMaterials = sorted([m for m in pair_mean_values_per_material.keys() if m not in orderedMaterials])
materialsSeq = orderedMaterials + remainingMaterials
labels = materialsSeq + ['Average']

fig, axes = plt.subplots(3, 2, figsize=(12, 15))
axes = axes.flatten()

for idx, metric in enumerate(['MRAE', 'RMSE', 'SAM', 'FID', 'PSNR', 'SSIM']):
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

    x = np.arange(len(labels_with_gap))
    colors = ['tab:blue'] * len(sortedVals) + ['none'] + ['tab:orange']
    dataMaterials = np.array(sortedVals, dtype=float)
    finiteMask = np.isfinite(dataMaterials)
    if np.any(finiteMask):
        arrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM', 'FID'] else '↑'
        if arrow == '↓':
            bestIdx = int(np.nanargmin(dataMaterials))
            worstIdx = int(np.nanargmax(dataMaterials))
        else:
            bestIdx = int(np.nanargmax(dataMaterials))
            worstIdx = int(np.nanargmin(dataMaterials))
        colors[bestIdx] = 'tab:green'
        colors[worstIdx] = 'tab:red'

    bars = ax.bar(x, data, color=colors)
    titleArrow = '↓' if metric in ['MRAE', 'RMSE', 'SAM', 'FID'] else '↑'
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
