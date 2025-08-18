# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# load all_datas_dict from npy
all_datas_dict = np.load('/app/results/202508180653/eval/all_results.npy', allow_pickle=True)
# %%
all_datas_dict_norm = {}
for value in all_datas_dict:
    all_datas_dict_norm[value['sample_name']] = {
        'pred': (value['gt_matched'] - value['gt_matched'].min()) / (value['gt_matched'].max() - value['gt_matched'].min()),
        'gt': (value['hsi_gt'] - value['hsi_gt'].min()) / (value['hsi_gt'].max() - value['hsi_gt'].min()),
        'diff': np.abs(value['gt_matched'] - value['hsi_gt'])
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
JSON_PATH = '/app/results/202508180653/eval/summary.json'
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
fig, axes = plt.subplots(4, 3, figsize=(10, 14))
axes = axes.flatten()

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
    pred_mean = pred.mean(axis=(1, 2)) if pred.ndim == 3 else pred.mean(axis=0)
    gt_mean = gt.mean(axis=(1, 2)) if gt.ndim == 3 else gt.mean(axis=0)

    # 표준편차(스펙트럼의 변동성)도 계산
    pred_std = pred.std(axis=(1, 2)) if pred.ndim == 3 else pred.std(axis=0)
    gt_std = gt.std(axis=(1, 2)) if gt.ndim == 3 else gt.std(axis=0)

    x = np.arange(len(pred_mean))
    # # GT(갈색), Pred(파랑) + 표준편차 영역
    ax.plot(x, gt_mean, label='GT', linewidth=3, color='orange')
    ax.plot(x, pred_mean, label='Pred', linewidth=3, linestyle='--', color='tab:blue')

    ax.set_title(material.replace('blockwood', 'wood_block')
                          .replace('columnwood', 'wood_column')
                          .replace('plywood', 'wood_plywood')
                          .replace('clearglass', 'glass_clear')
                          .replace('matteglass', 'glass_matte'), fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])

# y축, x축 라벨은 왼쪽 아래만
axes[8].set_yticks([])
axes[8].set_xticks([])
axes[8].set_yticklabels([], fontsize=10)

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

        # 에러 맵 계산 (절대값 기준)
        error_map = np.abs(pred - gt)
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
