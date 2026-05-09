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

detailed_pattern = r'actor_loss:\s*([-\d.]+)\s*±\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]\s+critic_loss:\s*([-\d.]+)\s*±\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]\s+reward:\s*([-\d.]+)\s*±\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]\s+r1:\s*([-\d.]+)\s*±\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]\s+r2:\s*([-\d.]+)\s*±\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]\s+r3:\s*([-\d.]+)\s*±\s*([-\d.]+)\s*\[([-\d.]+),\s*([-\d.]+)\]'

epochs = []
actor_losses, actor_stds, actor_lows, actor_highs = [], [], [], []
critic_losses, critic_stds, critic_lows, critic_highs = [], [], [], []
rewards, reward_stds, reward_lows, reward_highs = [], [], [], []
r1s, r1_stds, r1_lows, r1_highs = [], [], [], []
r2s, r2_stds, r2_lows, r2_highs = [], [], [], []
r3s, r3_stds, r3_lows, r3_highs = [], [], [], []

epoch_matches = list(re.finditer(epoch_pattern, log_content))
detailed_matches = list(re.finditer(detailed_pattern, log_content))

print(f"Found {len(epoch_matches)} epochs with main pattern")
print(f"Found {len(detailed_matches)} epochs with detailed pattern (with std and bounds)")

for i, match in enumerate(epoch_matches):
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

for i, match in enumerate(detailed_matches):
    if i < len(epochs):
        actor_stds.append(float(match.group(2)))
        actor_lows.append(float(match.group(3)))
        actor_highs.append(float(match.group(4)))
        critic_stds.append(float(match.group(6)))
        critic_lows.append(float(match.group(7)))
        critic_highs.append(float(match.group(8)))
        reward_stds.append(float(match.group(10)))
        reward_lows.append(float(match.group(11)))
        reward_highs.append(float(match.group(12)))
        r1_stds.append(float(match.group(14)))
        r1_lows.append(float(match.group(15)))
        r1_highs.append(float(match.group(16)))
        r2_stds.append(float(match.group(18)))
        r2_lows.append(float(match.group(19)))
        r2_highs.append(float(match.group(20)))
        r3_stds.append(float(match.group(22)))
        r3_lows.append(float(match.group(23)))
        r3_highs.append(float(match.group(24)))

while len(actor_stds) < len(epochs):
    actor_stds.append(0.0)
    actor_lows.append(0.0)
    actor_highs.append(0.0)
    critic_stds.append(0.0)
    critic_lows.append(0.0)
    critic_highs.append(0.0)
    reward_stds.append(0.0)
    reward_lows.append(0.0)
    reward_highs.append(0.0)
    r1_stds.append(0.0)
    r1_lows.append(0.0)
    r1_highs.append(0.0)
    r2_stds.append(0.0)
    r2_lows.append(0.0)
    r2_highs.append(0.0)
    r3_stds.append(0.0)
    r3_lows.append(0.0)
    r3_highs.append(0.0)

csv_file = os.path.join(output_dir, "training_stats_detailed.csv")
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'epoch', 
        'actor_loss', 'actor_std', 'actor_lower', 'actor_upper',
        'critic_loss', 'critic_std', 'critic_lower', 'critic_upper',
        'reward', 'reward_std', 'reward_lower', 'reward_upper',
        'r1', 'r1_std', 'r1_lower', 'r1_upper',
        'r2', 'r2_std', 'r2_lower', 'r2_upper',
        'r3', 'r3_std', 'r3_lower', 'r3_upper'
    ])
    for i in range(len(epochs)):
        writer.writerow([
            epochs[i],
            actor_losses[i], actor_stds[i], actor_lows[i], actor_highs[i],
            critic_losses[i], critic_stds[i], critic_lows[i], critic_highs[i],
            rewards[i], reward_stds[i], reward_lows[i], reward_highs[i],
            r1s[i], r1_stds[i], r1_lows[i], r1_highs[i],
            r2s[i], r2_stds[i], r2_lows[i], r2_highs[i],
            r3s[i], r3_stds[i], r3_lows[i], r3_highs[i]
        ])

print(f"Detailed CSV saved to: {csv_file}")
print(f"Total detailed epochs: {len(detailed_matches)}")

fig, axes = plt.subplots(3, 2, figsize=(14, 12))
fig.suptitle('Training Stats with Std & Confidence Intervals - Job 167832251', fontsize=14, fontweight='bold')

ax = axes[0, 0]
ax.plot(epochs, actor_losses, 'b-', linewidth=1.5, marker='o', markersize=2, label='Mean')
ax.fill_between(epochs, actor_lows, actor_highs, alpha=0.3, color='blue', label='CI [mean±std]')
ax.set_xlabel('Epoch')
ax.set_ylabel('Actor Loss')
ax.set_title('Actor Loss over Epochs')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(epochs, critic_losses, 'r-', linewidth=1.5, marker='o', markersize=2, label='Mean')
ax.fill_between(epochs, critic_lows, critic_highs, alpha=0.3, color='red', label='CI [mean±std]')
ax.set_xlabel('Epoch')
ax.set_ylabel('Critic Loss')
ax.set_title('Critic Loss over Epochs')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(epochs, rewards, 'g-', linewidth=1.5, marker='o', markersize=2, label='Mean')
ax.fill_between(epochs, reward_lows, reward_highs, alpha=0.3, color='green', label='CI [mean±std]')
ax.set_xlabel('Epoch')
ax.set_ylabel('Reward')
ax.set_title('Reward over Epochs')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(epochs, r1s, 'm-', linewidth=1.5, marker='o', markersize=2, label='r1')
ax.fill_between(epochs, r1_lows, r1_highs, alpha=0.2, color='magenta')
ax.plot(epochs, r2s, 'c-', linewidth=1.5, marker='o', markersize=2, label='r2')
ax.fill_between(epochs, r2_lows, r2_highs, alpha=0.2, color='cyan')
ax.plot(epochs, r3s, 'y-', linewidth=1.5, marker='o', markersize=2, label='r3')
ax.fill_between(epochs, r3_lows, r3_highs, alpha=0.2, color='yellow')
ax.set_xlabel('Epoch')
ax.set_ylabel('Value')
ax.set_title('r1, r2, r3 over Epochs')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 0]
ax.plot(epochs, actor_stds, 'b-', linewidth=1.5, marker='o', markersize=2, label='Actor Std')
ax.plot(epochs, critic_stds, 'r-', linewidth=1.5, marker='o', markersize=2, label='Critic Std')
ax.set_xlabel('Epoch')
ax.set_ylabel('Std')
ax.set_title('Std Comparison over Epochs')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.plot(epochs, reward_stds, 'g-', linewidth=1.5, marker='o', markersize=2, label='Reward Std')
ax.plot(epochs, r1_stds, 'm-', linewidth=1.5, marker='o', markersize=2, label='r1 Std')
ax.plot(epochs, r2_stds, 'c-', linewidth=1.5, marker='o', markersize=2, label='r2 Std')
ax.set_xlabel('Epoch')
ax.set_ylabel('Std')
ax.set_title('Reward & Component Stds over Epochs')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = os.path.join(output_dir, "training_plots_detailed.png")
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"Detailed plots saved to: {plot_file}")

fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(epochs, rewards, 'g-', linewidth=2, marker='o', markersize=3, label='Reward Mean')
ax2.fill_between(epochs, reward_lows, reward_highs, alpha=0.3, color='green', label='95% CI')
reward_max_idx = rewards.index(max(rewards))
ax2.scatter([epochs[reward_max_idx]], [rewards[reward_max_idx]], color='red', s=100, zorder=5)
ax2.annotate(f"Max: {max(rewards):.4f}\n@ epoch {epochs[reward_max_idx]}", 
            xy=(epochs[reward_max_idx], rewards[reward_max_idx]),
            xytext=(epochs[reward_max_idx]-15, rewards[reward_max_idx]+0.3),
            fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Reward', fontsize=12)
ax2.set_title('Reward Progression with Confidence Intervals - Job 167832251', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
plot_file2 = os.path.join(output_dir, "reward_with_ci.png")
plt.savefig(plot_file2, dpi=150, bbox_inches='tight')
print(f"Reward with CI plot saved to: {plot_file2}")

fig3, axes3 = plt.subplots(2, 3, figsize=(15, 8))
fig3.suptitle('Component Values with Confidence Intervals - Job 167832251', fontsize=14, fontweight='bold')

for idx, (data, stds, lows, highs, name, color) in enumerate([
    (actor_losses, actor_stds, actor_lows, actor_highs, 'Actor Loss', 'blue'),
    (critic_losses, critic_stds, critic_lows, critic_highs, 'Critic Loss', 'red'),
    (rewards, reward_stds, reward_lows, reward_highs, 'Reward', 'green'),
    (r1s, r1_stds, r1_lows, r1_highs, 'r1', 'magenta'),
    (r2s, r2_stds, r2_lows, r2_highs, 'r2', 'cyan'),
    (r3s, r3_stds, r3_lows, r3_highs, 'r3', 'yellow')
]):
    row = idx // 3
    col = idx % 3
    ax = axes3[row, col]
    ax.plot(epochs, data, f'{color[0]}-', linewidth=1.5, marker='o', markersize=2, label='Mean')
    ax.fill_between(epochs, lows, highs, alpha=0.3, color=color, label='95% CI')
    ax.set_xlabel('Epoch')
    ax.set_ylabel(name)
    ax.set_title(f'{name} over Epochs')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plot_file3 = os.path.join(output_dir, "all_components_with_ci.png")
plt.savefig(plot_file3, dpi=150, bbox_inches='tight')
print(f"All components plot saved to: {plot_file3}")

print(f"\n=== Summary ===")
print(f"Total epochs: {len(epochs)}")
print(f"Best reward: {max(rewards):.4f} @ epoch {epochs[rewards.index(max(rewards))]}")
print(f"Best actor loss: {min(actor_losses):.4f} @ epoch {epochs[actor_losses.index(min(actor_losses))]}")
print(f"Best critic loss: {min(critic_losses):.4f} @ epoch {epochs[critic_losses.index(min(critic_losses))]}")