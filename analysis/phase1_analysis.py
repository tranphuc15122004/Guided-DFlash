#!/usr/bin/env python3
"""
Phase 1 Training Analysis — NegativeLogitPredictor_Dense

Generates plots and statistical summary from training_history.csv.
Run from repo root: python analysis/phase1_analysis.py
"""

import os
import sys
import json
import math
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Paths ──
REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints" / "phase1_169125108.gadi-pbs_neg"
CSV_PATH = CKPT / "training_history.csv"
CONFIG_PATH = CKPT / "config.json"
OUTPUT_DIR = CKPT / "analysis_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──
with open(CONFIG_PATH) as f:
    config = json.load(f)

rows = []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

epochs = [int(r['epoch']) for r in rows]
n_epochs = len(epochs)

def get_col(name):
    return np.array([float(r[name]) for r in rows], dtype=np.float64)

# Extract all metric columns
loss = get_col('loss_mean')
ce_loss = get_col('ce_loss_mean')
bce_loss = get_col('bce_loss_mean')
top1 = get_col('top1_acc_mean')
top5 = get_col('top5_acc_mean')
top10 = get_col('top10_acc_mean')
mean_rank = get_col('mean_rank_mean')
median_rank = get_col('median_rank_mean')
rank_std = get_col('rank_std_mean')
bce_acc = get_col('bce_acc_mean')
cd_entropy = get_col('cd_entropy_mean')
entropy_red = get_col('entropy_reduction_mean')
pos_neg_kl = get_col('pos_neg_kl_mean')
target_cd_score = get_col('target_cd_score_mean')
action_sep = get_col('action_separation_mean')
action_mag = get_col('action_magnitude_mean')
lr_vals = get_col('lr')

# Find best epoch for top1
best_idx = int(np.argmax(top1))

# ═══════════════════════════════════════════════════════════
#  Plot 1: Loss Curves (CE + BCE + Total)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, loss, label='Total Loss', color='#2c3e50', linewidth=1.5)
ax.plot(epochs, ce_loss, label='CE Loss (CD cross-entropy)', color='#e74c3c', linewidth=1.2)
ax.plot(epochs, bce_loss, label='BCE Loss (auxiliary)', color='#3498db', linewidth=1.2)
ax.axvline(epochs[best_idx], color='gray', linestyle='--', alpha=0.5, label=f'Best top-1 (epoch {epochs[best_idx]})')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Phase 1 Training Losses', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'loss_curves.png', dpi=150)
plt.close(fig)
print("  ✓ loss_curves.png")

# ═══════════════════════════════════════════════════════════
#  Plot 2: Accuracy Metrics (Top-1, Top-5, Top-10)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, top1 * 100, label='Top-1 Accuracy', color='#27ae60', linewidth=1.5)
ax.plot(epochs, top5 * 100, label='Top-5 Accuracy', color='#2980b9', linewidth=1.2)
ax.plot(epochs, top10 * 100, label='Top-10 Accuracy', color='#8e44ad', linewidth=1.2)
ax.axhline(top1[0] * 100, color='#27ae60', linestyle=':', alpha=0.4, label=f'Initial Top-1 ({top1[0]*100:.2f}%)')
ax.axhline(top1[-1] * 100, color='#27ae60', linestyle='--', alpha=0.6, label=f'Final Top-1 ({top1[-1]*100:.2f}%)')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Token Prediction Accuracy within Top-K', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'accuracy_curves.png', dpi=150)
plt.close(fig)
print("  ✓ accuracy_curves.png")

# ═══════════════════════════════════════════════════════════
#  Plot 3: Rank Metrics
# ═══════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 5))
color1, color2, color3 = '#e67e22', '#c0392b', '#7f8c8d'
ax1.plot(epochs, mean_rank, label='Mean Rank', color=color1, linewidth=1.5)
ax1.plot(epochs, rank_std, label='Rank Std', color=color2, linewidth=1.2, linestyle='--')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Rank Value', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

# Median rank on secondary axis (it's always 1, so just annotate)
ax1.annotate('Median Rank = 1.0 (constant)',
             xy=(n_epochs * 0.6, 1.5), fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

ax1.set_title('Target Token Rank Statistics', fontsize=14, fontweight='bold')
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, fontsize=10)
ax1.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'rank_metrics.png', dpi=150)
plt.close(fig)
print("  ✓ rank_metrics.png")

# ═══════════════════════════════════════════════════════════
#  Plot 4: CD Distribution Metrics
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, cd_entropy, label='CD Entropy', color='#1abc9c', linewidth=1.5)
ax.plot(epochs, entropy_red, label='Entropy Reduction (CD - pos)', color='#e84393', linewidth=1.2)
ax.plot(epochs, pos_neg_kl, label='KL(Pos || Neg Orig)', color='#f39c12', linewidth=1.2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Distribution Metrics — Entropy & Divergence', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'distribution_metrics.png', dpi=150)
plt.close(fig)
print("  ✓ distribution_metrics.png")

# ═══════════════════════════════════════════════════════════
#  Plot 5: Action Metrics (Separation & Magnitude)
# ═══════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 5))
color1, color2 = '#9b59b6', '#2ecc71'
ax1.plot(epochs, action_sep, label='Action Separation', color=color1, linewidth=1.5)
ax1.axhline(0, color='gray', linestyle='-', alpha=0.3)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Action Separation', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(epochs, action_mag, label='Action Magnitude', color=color2, linewidth=1.5, linestyle='--')
ax2.set_ylabel('Action Magnitude', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='lower right')
ax1.set_title('Negative Predictor Action Statistics', fontsize=14, fontweight='bold')
ax1.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'action_metrics.png', dpi=150)
plt.close(fig)
print("  ✓ action_metrics.png")

# ═══════════════════════════════════════════════════════════
#  Plot 6: BCE Accuracy & Loss
# ═══════════════════════════════════════════════════════════
fig, ax1 = plt.subplots(figsize=(10, 5))
color1, color2 = '#e74c3c', '#2c3e50'
ax1.plot(epochs, bce_loss, label='BCE Loss', color=color1, linewidth=1.5)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('BCE Loss', fontsize=12, color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(epochs, bce_acc * 100, label='BCE Accuracy (%)', color=color2, linewidth=1.5, linestyle='--')
ax2.set_ylabel('BCE Accuracy (%)', fontsize=12, color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc='center right')
ax1.set_title('BCE Auxiliary: Loss & Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'bce_metrics.png', dpi=150)
plt.close(fig)
print("  ✓ bce_metrics.png")

# ═══════════════════════════════════════════════════════════
#  Plot 7: Learning Rate Schedule
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(epochs, lr_vals, color='#e74c3c', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate', fontsize=12)
ax.set_title('Cosine LR Schedule (3e-4 → 1e-5)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'lr_schedule.png', dpi=150)
plt.close(fig)
print("  ✓ lr_schedule.png")

# ═══════════════════════════════════════════════════════════
#  Plot 8: Target CD Score (how well target ranks in CD distribution)
# ═══════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, target_cd_score, label='Target CD Score', color='#d35400', linewidth=1.5)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('CD Score (logit value of target token)', fontsize=12)
ax.set_title('Target Token Score in Contrastive Decoding Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, n_epochs)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'target_cd_score.png', dpi=150)
plt.close(fig)
print("  ✓ target_cd_score.png")

# ═══════════════════════════════════════════════════════════
#  Summary statistics
# ═══════════════════════════════════════════════════════════
print("\n" + "═" * 65)
print("  PHASE 1 TRAINING SUMMARY")
print("═" * 65)

print(f"\n  Config:")
print(f"    Model:           {config.get('dense', False) and 'NegativeLogitPredictor_Dense' or 'NegativeLogitPredictor'}")
print(f"    Dataset:         {config['data_path']}")
print(f"    Epochs planned:  {config['epochs']}")
print(f"    Epochs completed:{n_epochs}")
print(f"    Batch size:      {config['batch_size']}")
print(f"    LR:              {config['lr']} → {config['lr_min']} ({config['lr_scheduler']})")
print(f"    Predict delta:   {config['predict_delta']}")
print(f"    Lambda BCE:      {config['lambda_bce']}")
print(f"    Early boost:     first {config['early_boost_n']} tokens × {config['early_boost_weight']}")

print(f"\n  Best epoch (top-1): epoch {epochs[best_idx]}")
print(f"  Best top-1:        {top1[best_idx]*100:.2f}%")
print(f"  Final top-5:       {top5[-1]*100:.2f}%")
print(f"  Final top-10:      {top10[-1]*100:.2f}%")

print(f"\n  Loss improvement:")
print(f"    Total:    {loss[0]:.4f} → {loss[-1]:.4f}  ({loss[0]-loss[-1]:.4f} Δ, {(1-loss[-1]/loss[0])*100:.1f}% drop)")
print(f"    CE:       {ce_loss[0]:.4f} → {ce_loss[-1]:.4f}  ({ce_loss[0]-ce_loss[-1]:.4f} Δ)")
print(f"    BCE:      {bce_loss[0]:.4f} → {bce_loss[-1]:.4f}  ({bce_loss[0]-bce_loss[-1]:.4f} Δ)")

print(f"\n  Accuracy improvement:")
print(f"    Top-1:    {top1[0]*100:.2f}% → {top1[-1]*100:.2f}%  (Δ {(top1[-1]-top1[0])*100:.2f}pp)")
print(f"    Top-5:    {top5[0]*100:.2f}% → {top5[-1]*100:.2f}%  (Δ {(top5[-1]-top5[0])*100:.2f}pp)")
print(f"    Top-10:   {top10[0]*100:.2f}% → {top10[-1]*100:.2f}%  (Δ {(top10[-1]-top10[0])*100:.2f}pp)")
print(f"    BCE Acc:  {bce_acc[0]*100:.2f}% → {bce_acc[-1]*100:.2f}%  (Δ {(bce_acc[-1]-bce_acc[0])*100:.2f}pp)")

print(f"\n  Rank improvement:")
print(f"    Mean:     {mean_rank[0]:.2f} → {mean_rank[-1]:.2f}  (Δ {mean_rank[0]-mean_rank[-1]:.2f})")
print(f"    Median:   {median_rank[0]:.0f} (always 1.0 — target is usually #1)")
print(f"    Std:      {rank_std[0]:.2f} → {rank_std[-1]:.2f}")

print(f"\n  Distribution / CD quality:")
print(f"    CD Entropy:       {cd_entropy[0]:.3f} → {cd_entropy[-1]:.3f}")
print(f"    Entropy Red.:     {entropy_red[0]:.4f} → {entropy_red[-1]:.4f}")
print(f"    KL(Pos||Neg):     {pos_neg_kl[0]:.3f} → {pos_neg_kl[-1]:.3f}  (↑ {pos_neg_kl[-1]-pos_neg_kl[0]:.2f})")
print(f"    Target CD Score:  {target_cd_score[0]:.3f} → {target_cd_score[-1]:.3f}  (↑ {target_cd_score[-1]-target_cd_score[0]:.2f})")

print(f"\n  Action stats:")
print(f"    Separation:  {action_sep[0]:.3f} → {action_sep[-1]:.3f}")
print(f"    Magnitude:   {action_mag[0]:.3f} → {action_mag[-1]:.3f}")

print(f"\n  Plots saved to: {OUTPUT_DIR}")
print(f"\n  Done.")
