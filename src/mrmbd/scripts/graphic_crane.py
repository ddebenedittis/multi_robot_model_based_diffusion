"""Crane pendulum diffusion animation: visualize reverse diffusion steps."""

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
l = 1
dt = 0.01

# Load states saved during reverse diffusion
data = np.load("results/latest-crane/crane_states_over_steps.npz", allow_pickle=True)
states = np.asarray(data["states"])  # shape: (Nsteps, Nplot, H+1, 4)
Nsteps, Nplot, Hplus1, _ = states.shape

# Precompute (x, y) of the suspended mass
all_trajs = np.zeros((Nsteps, Nplot, Hplus1, 2))
for step in range(Nsteps):
    for i in range(Nplot):
        theta = states[step, i, :, 0]
        x = states[step, i, :, 2]
        x_l = x + l * np.sin(theta)
        y_l = l * np.cos(theta)
        all_trajs[step, i] = np.stack([x_l, y_l], axis=-1)

# Setup figure
fig, ax = plt.subplots(figsize=(6, 4))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_aspect("equal")
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")

colors = plt.cm.plasma(np.linspace(0, 1, Nplot))
lines = [ax.plot([], [], "-", color=colors[i], alpha=0.7)[0] for i in range(Nplot)]
title = ax.set_title("")


def init():
    for line in lines:
        line.set_data([], [])
    return lines


def update(step):
    title.set_text(f"Reverse diffusion step {step + 1}/{Nsteps}")
    step_trajs = all_trajs[step]
    for i in range(Nplot):
        lines[i].set_data(step_trajs[i, :, 0], step_trajs[i, :, 1])
    return lines


ani = FuncAnimation(fig, update, frames=Nsteps, init_func=init, interval=300, blit=False)

out_dir = "results/latest-crane"
os.makedirs(out_dir, exist_ok=True)
video_path = os.path.join(out_dir, "pendulum_diffusion_steps.mp4")
ani.save(video_path, fps=3)
print(f"Video saved: {video_path}")
