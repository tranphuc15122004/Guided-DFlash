import re
import csv
import matplotlib.pyplot as plt
import os

log_file = "train_alpha_h200.pbs.o167832251"
output_dir = "analysis/training_plots_167832251"

os.makedirs(output_dir, exist_ok=True)

with open(log_file, 'r') as f:
    log_content = f.read()

epoch_pattern = r'✔ Epoch\s+(\d+)\s+\|\s+actor_loss=([-\d.]+)\s+critic_loss=([-\d.]+)\s+reward=([-\d.]+)\s+r1=([-\d.]+)\s+r2=([-\d.]+)\s+r3=([-\d.]+)'

epochs = []
actor_losses = []
critic_losses = []
rewards = []
r1s = []
r2s = []
r3s = []

epoch_matches = list(re.finditer(epoch_pattern, log_content))

for match in epoch_matches:
    epoch = int(match.group(1))
    actor_loss = float(match.group(2))
    critic_loss = float(match.group(3))
    reward = float(match.group(4))
    r1 = float(match.group(5))
    r2 = float(match.group(6))
    r3 = float(match.group(7))
    
    epochs.append(epoch)
    actor_losses.append(actor_loss)
    critic_losses.append(critic_loss)
    rewards.append(reward)
    r1s.append(r1)
    r2s.append(r2)
    r3s.append(r3)

csv_file = os.path.join(output_dir, "training_stats.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'actor_loss', 'critic_loss', 'reward', 'r1', 'r2', 'r3'])
    for i in range(len(epochs)):
        writer.writerow([
            epochs[i],
            actor_losses[i],
            critic_losses[i],
            rewards[i],
            r1s[i],
            r2s[i],
            r3s[i]
        ])

print(f"CSV saved to: {csv_file}")
print(f"Total epochs: {len(epochs)}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Training Stats - Job 167832251 (Epochs 1-70)', fontsize=14, fontweight='bold')

axes[0, 0].plot(epochs, actor_losses, 'b-', linewidth=1.5, marker='o', markersize=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Actor Loss')
axes[0, 0].set_title('Actor Loss over Epochs')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(epochs, critic_losses, 'r-', linewidth=1.5, marker='o', markersize=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Critic Loss')
axes[0, 1].set_title('Critic Loss over Epochs')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(epochs, rewards, 'g-', linewidth=1.5, marker='o', markersize=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Reward')
axes[1, 0].set_title('Reward over Epochs')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(epochs, r1s, 'm-', label='r1', linewidth=1.5, marker='o', markersize=2)
axes[1, 1].plot(epochs, r2s, 'c-', label='r2', linewidth=1.5, marker='o', markersize=2)
axes[1, 1].plot(epochs, r3s, 'y-', label='r3', linewidth=1.5, marker='o', markersize=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('r1, r2, r3 over Epochs')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(output_dir, "training_plots.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Plots saved to: {plot_file}")

fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(epochs, rewards, 'g-', linewidth=2, marker='o', markersize=3, label='Reward')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.set_title('Reward Progression - Job 167832251', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()
reward_max_idx = rewards.index(max(rewards))
ax2.scatter([epochs[reward_max_idx]], [rewards[reward_max_idx]], color='red', s=100, zorder=5, label=f'Max: {max(rewards):.4f} @ epoch {epochs[reward_max_idx]}')
ax2.legend()
plot_file2 = os.path.join(output_dir, "reward_plot.png")
plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
print(f"Reward plot saved to: {plot_file2}")