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
import json
JSON_PATH = '/app/results/align_202508071057_202508070233_epoch_2000/eval/summary.json'
with open(JSON_PATH, 'r') as f:
    results = json.load(f)

# %%

# %%

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
print(MATERIALS[0], MATERIALS[1], values['PSNR'][0, 1], values['SSIM'][0, 1], values['RMSE'][0, 1], values['MRAE'][0, 1], values['SAM'][0, 1], values['FID'][0, 1])
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
            row = [material] + [f"{metrics[m]:.4f}" for m in metricOrder]
            rows.append(row)
        # 평균 행 계산 (선택된 materialsSeq 기준)
        if len(materialsSeq) > 0:
            avgVals = { m: float(np.nanmean([meanValuesDict[name][m] for name in materialsSeq])) for m in metricOrder }
            avgRow = ['Average'] + [f"{avgVals[m]:.4f}" for m in metricOrder]
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