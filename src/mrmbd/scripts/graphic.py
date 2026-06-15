"""Multi-car post-processing: analysis plots and global diffusion video."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tyro
from scipy.spatial.distance import pdist

from mrmbd.envs import MultiCar2d
from mrmbd.envs.multi_car import Args
from mrmbd.utils import plot_diffusion_cloud, set_plot_style

# Output directory
path = "results/latest-multicar"

# Academic plotting style
set_plot_style()

# Find the most recent optimized data file
candidates = [f for f in os.listdir(path) if f.startswith("optimized_data_") and f.endswith(".npz")]

if not candidates:
    raise FileNotFoundError("No optimized_data_*.npz file found in the directory.")

candidates = sorted(candidates, key=lambda f: os.path.getmtime(os.path.join(path, f)))
filename = candidates[-1]
print(f"Loaded most recent file: {filename}")

if "LIDEC" in filename:
    ecd_tag = "LIDEC"
elif "LID" in filename:
    ecd_tag = "LID"
else:
    ecd_tag = "unknown"

form = "form" if "form" in filename else "_"

data = np.load(os.path.join(path, filename))
traj = data["traj"]
goals = data["goals"]
rewards = data["rewards"]

n, T, _ = traj.shape

# 1. Minimum inter-robot distance over time
min_distances = [pdist(traj[:, t, :2]).min() for t in range(T)]
plt.figure()
plt.plot(min_distances, lw=1.2, color="#1f77b4")
plt.title("Minimum inter-robot distance over time")
plt.xlabel("Time [step]")
plt.ylabel("Minimum distance [m]")
plt.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{path}/plot_min_distance_{ecd_tag}_{form}.png")
plt.close()

# 2. Final error relative to goal
final_positions = traj[:, -1, :2]
goal_positions = goals[:, :2]
errors = np.linalg.norm(final_positions - goal_positions, axis=1)
plt.figure()
plt.bar(range(n), errors, color="#2ca02c")
plt.title("Final error relative to goal")
plt.xlabel("Robot")
plt.ylabel("Error [m]")
plt.grid(True, linestyle="-", alpha=0.6)
plt.tight_layout()
plt.savefig(f"{path}/plot_goal_errors_{ecd_tag}_{form}.png")
plt.close()

# 3. Mean reward per robot over iterations
K = rewards.shape[0]
plt.figure()
for i in range(n):
    plt.plot(range(K), rewards[:, i], label=f"Robot {i}")
plt.title("Mean reward per robot over iterations")
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.grid(True, linestyle="-", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"{path}/plot_rewards_{ecd_tag}_{form}.png")
plt.close()

# 4. Distance to goal over time
errors_over_time = np.zeros((n, T))
for i in range(n):
    for t in range(T):
        errors_over_time[i, t] = np.linalg.norm(traj[i, t, :2] - goal_positions[i])
plt.figure()
for i in range(n):
    plt.plot(errors_over_time[i], label=f"Robot {i}")
plt.title("Distance to goal over time")
plt.xlabel("Time [step]")
plt.ylabel("Error [m]")
plt.grid(True, linestyle="-", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig(f"{path}/plot_goal_error_over_time_{ecd_tag}_{form}.png")
plt.close()

# === Global diffusion cloud (static images + video) ===
args = tyro.cli(Args)
env = MultiCar2d(
    n=args.n_robots,
    formation_shift=args.formation_shift,
    ECD=args.ECD,
    obstacles_enabled=args.obstacles_enabled,
)

plot_diffusion_cloud(path, env)
