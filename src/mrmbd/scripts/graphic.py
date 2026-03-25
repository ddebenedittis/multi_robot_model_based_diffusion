"""Multi-car post-processing: analysis plots and global diffusion video."""

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import tyro
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist

from mrmbd.envs import MultiCar2d
from mrmbd.envs.multi_car import Args

# Output directory
path = "results/latest-multicar"

# Academic plotting style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["CMU Serif", "DejaVu Serif", "Times"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
    }
)

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

# === Global diffusion video ===
output_path = os.path.join(path, "global_diffusion_video.mp4")
os.makedirs(os.path.dirname(output_path), exist_ok=True)

data = np.load(os.path.join(path, "global_Yi_list.npz"))
trajectories_all = data["trajectories_samples"]
trajectories_denoised = data["trajectories_denoised"]

args = tyro.cli(Args)
env = MultiCar2d(
    n=args.n_robots,
    formation_shift=args.formation_shift,
    ECD=args.ECD,
    obstacles_enabled=args.obstacles_enabled,
)


def draw_obstacle_penalty_zones(ax, env):
    buffer_min, buffer_max = 0.2, 0.5
    for x_c, y_c, w, h in env.static_obstacles:
        rect = plt.Rectangle(
            (x_c - w / 2, y_c - h / 2),
            w,
            h,
            linewidth=1.0,
            edgecolor="black",
            facecolor="#d3d3d3",
            zorder=1,
        )
        ax.add_patch(rect)
        rect_outer = plt.Rectangle(
            (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
            w + 2 * buffer_max,
            h + 2 * buffer_max,
            linewidth=0.8,
            edgecolor="none",
            facecolor="#a6bddb",
            alpha=0.25,
            zorder=1,
        )
        ax.add_patch(rect_outer)
        rect_inner = plt.Rectangle(
            (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
            w + 2 * buffer_min,
            h + 2 * buffer_min,
            linewidth=0.8,
            edgecolor="none",
            facecolor="#3690c0",
            alpha=0.35,
            zorder=2,
        )
        ax.add_patch(rect_inner)


palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

n_plot = Args.n_robots
cmap = plt.get_cmap("tab20", n_plot)
fig, ax = plt.subplots(figsize=(6, 6))


def update(frame):
    ax.clear()
    ax.set_title(f"Reverse Diffusion Step {frame}")
    ax.set_aspect("equal")
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.grid(True, linestyle="-", color="k", linewidth=0.6, alpha=0.7)
    draw_obstacle_penalty_zones(ax, env)

    samples = trajectories_all[frame]
    for i in range(n_plot):
        for s in range(min(80, samples.shape[0])):
            traj = samples[s]
            ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.07, color=palette[i])

    traj_opt = trajectories_denoised[frame]
    for i in range(n_plot):
        ax.plot(traj_opt[:, i, 0], traj_opt[:, i, 1], color=palette[i], linewidth=2)
        ax.plot(traj_opt[0, i, 0], traj_opt[0, i, 1], "o", color=palette[i], markersize=4)

    return []


ani = animation.FuncAnimation(fig, update, frames=len(trajectories_all), interval=150)


def save_static_diffusion_images(trajectories_all, trajectories_denoised, env, path, cmap):
    """Save static images for initial and final reverse diffusion steps."""
    steps = {"initial": 0, "final": len(trajectories_all) - 1}

    for tag, idx in steps.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.grid(True, linestyle="-", color="k", linewidth=0.6, alpha=0.7)
        draw_obstacle_penalty_zones(ax, env)

        n = env.n
        samples = trajectories_all[idx]
        for i in range(n):
            for s in range(min(80, samples.shape[0])):
                traj = samples[s]
                ax.plot(traj[:, i, 0], traj[:, i, 1], alpha=0.07, color=palette[i])

        traj_opt = trajectories_denoised[idx]
        for i in range(n):
            gx, gy = env.xg[i, 0], env.xg[i, 1]
            ax.plot(
                gx, gy, marker="s", color=palette[i], markersize=5.5, markeredgewidth=0.8, zorder=5
            )
            ax.plot(traj_opt[:, i, 0], traj_opt[:, i, 1], color=palette[i], linewidth=2)
            ax.plot(traj_opt[0, i, 0], traj_opt[0, i, 1], "o", color=palette[i], markersize=4)

        ax.set_title(f"Reverse Diffusion - Step {tag}", pad=6)
        legend_elements = [
            Line2D([0], [0], color="gray", lw=1, alpha=0.3, label="Stochastic samples"),
            Line2D([0], [0], color="k", lw=2, label="Optimized trajectory"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="k",
                markersize=5,
                label="Initial position",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="k",
                markersize=5,
                label="Final position",
            ),
        ]
        ax.legend(handles=legend_elements, loc="upper right", frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"global_diffusion_{tag}.png"), dpi=300)
        plt.close(fig)

    print("Static images saved: global_diffusion_initial.png, global_diffusion_final.png")


save_static_diffusion_images(trajectories_all, trajectories_denoised, env, path, cmap)

ani.save(output_path, fps=5, dpi=150)
print(f"Video saved: {output_path}")
