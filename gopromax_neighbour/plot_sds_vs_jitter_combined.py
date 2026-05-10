"""
Combined SDS Score vs. Jitter Distance Plot
=============================================

Reads multiple *_sds_vs_jitter.csv files and overlays them on a single plot
with distinct colours.

Usage
-----
    python plot_sds_vs_jitter_combined.py
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

# ---- Data sources -----------------------------------------------------------
SOURCES = [
    {
        'csv': '../gaussian-splatting/output/run_01/sds_plot/0004_back_sds_vs_jitter.csv',
        'label': '3DGS baseline',
        'color': 'C0',
        'meters_per_unit': 2.4,  # per-source x-axis scale (world units → metres)
    },
    {
        'csv': 'output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune/gopromax_neighbour/sky_mask_v1/sds_plot/0004_back_sds_vs_jitter.csv',
        'label': 'DA2 loss (ours)',
        'color': 'C1',
        'meters_per_unit': 7.0,
    },
    {
        'csv': 'output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/gopromax_neighbour/sky_mask_v1_gan/sds_plot/0004_back_sds_vs_jitter.csv',
        'label': 'DA2 loss + critic (ours)',
        'color': 'C2',
        'meters_per_unit': 7.0,
    },
]

OUTPUT_PATH = 'output/combined_sds_vs_jitter_0004_back.png'
METERS_PER_UNIT = 7.0   # fallback scale if 'meters_per_unit' not set per source
ERRORBAR_STYLE = 'band'  # 'band', 'errorbar', or 'both'

# ---- Plot -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))

for src in SOURCES:
    csv_path = os.path.join(os.path.dirname(__file__), src['csv'])
    distances, means, stds = [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            distances.append(float(row['jitter_distance']))
            means.append(float(row['sds_mean']))
            stds.append(float(row['sds_std']))

    distances_m = np.array(distances) * src.get('meters_per_unit', METERS_PER_UNIT)
    means = np.array(means)
    stds  = np.array(stds)

    ax.plot(distances_m, means, marker='o', linewidth=1.5, markersize=5,
            color=src['color'], label=src['label'])

    if ERRORBAR_STYLE in ('band', 'both'):
        ax.fill_between(distances_m, means - stds, means + stds,
                        alpha=0.2, color=src['color'])
    if ERRORBAR_STYLE in ('errorbar', 'both'):
        ax.errorbar(distances_m, means, yerr=stds,
                    fmt='none', ecolor=src['color'], capsize=3, alpha=0.8)

ax.set_xlabel('Jitter distance (m)')
ax.set_ylabel('SDS score (lower = more realistic)')
ax.set_title('SDS score vs. lateral jitter — 0004_back')
ax.legend(loc='best', fontsize=9)
ax.grid(True, linestyle='--', alpha=0.5)
fig.tight_layout()

os.makedirs(os.path.dirname(os.path.join(os.path.dirname(__file__), OUTPUT_PATH)), exist_ok=True)
out_path = os.path.join(os.path.dirname(__file__), OUTPUT_PATH)
fig.savefig(out_path, dpi=150)
print(f"Plot saved to {out_path}")
plt.close(fig)
