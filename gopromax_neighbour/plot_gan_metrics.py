"""
Visualize GAN training and evaluation metrics.

Usage
-----
    python plot_gan_metrics.py
    python plot_gan_metrics.py --run_dir output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2/gopromax_neighbour/sky_mask_v1_gan
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN = os.path.join(
    BASE,
    'output/22_300_da2loss_0.5_skymodel_1_0.01_0.5_tune_critic_100_v2'
    '/gopromax_neighbour/sky_mask_v1_gan',
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', default=DEFAULT_RUN)
    parser.add_argument('--output', default=None,
                        help='Save figure to this path instead of showing it')
    args = parser.parse_args()

    train_csv = os.path.join(args.run_dir, 'train_metrics.csv')
    eval_csv  = os.path.join(args.run_dir, 'eval_metrics.csv')

    train = pd.read_csv(train_csv)
    eval_ = pd.read_csv(eval_csv)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('GAN Training & Evaluation Metrics', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- 1. Critic / Generator losses ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train['epoch'], train['loss_critic'], label='Critic loss')
    ax1.plot(train['epoch'], train['loss_gen'],    label='Generator loss')
    ax1.set_title('Critic & Generator Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- 2. Reconstruction loss ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train['epoch'], train['loss_recon'], color='tab:orange')
    ax2.set_title('Reconstruction Loss (train)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)

    # --- 3. Wasserstein distance ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(train['epoch'], train['wasserstein_dist'], color='tab:purple')
    ax3.set_title('Wasserstein Distance')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Distance')
    ax3.grid(True, alpha=0.3)

    # --- 4. Train PSNR on-road vs off-road ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(train['epoch'], train['psnr_onroad'],  label='On-road')
    ax4.plot(train['epoch'], train['psnr_offroad'], label='Off-road')
    ax4.set_title('Train PSNR (on-road vs off-road)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('PSNR (dB)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # --- 5. Eval PSNR & L1 ---
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(eval_['epoch'], eval_['psnr'],    color='tab:blue',   label='PSNR (dB)')
    ax5_r = ax5.twinx()
    ax5_r.plot(eval_['epoch'], eval_['l1_loss'], color='tab:red', linestyle='--', label='L1 loss')
    ax5.set_title('Eval PSNR & L1 Loss')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('PSNR (dB)', color='tab:blue')
    ax5_r.set_ylabel('L1 Loss', color='tab:red')
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_r.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax5.grid(True, alpha=0.3)

    # --- 6. Eval SSIM ---
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(eval_['epoch'], eval_['ssim'], color='tab:green')
    ax6.set_title('Eval SSIM')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('SSIM')
    ax6.grid(True, alpha=0.3)

    out_path = args.output or os.path.join(args.run_dir, 'metrics_plot.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {out_path}")
    plt.show()


if __name__ == '__main__':
    main()
