"""Sweep PID gains for MBD on a given environment."""

import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

mbd_inner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbd")
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(os.path.dirname(__file__))]
sys.path.insert(0, mbd_inner)

import argparse
import csv
from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(f"JAX backend: {jax.default_backend()}")

from mbd.planners.mbd_planner import Args as MBDArgs
from mbd.planners.mbd_planner import run_diffusion

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="ant", help="Environment name (ant, halfcheetah, ...)")
parser.add_argument("--runs", type=int, default=3, help="Number of runs per config")
parser.add_argument("--steps", type=int, default=None, help="Number of diffusion steps (Ndiffuse)")
parser.add_argument("--samples", type=int, default=None, help="Number of samples (Nsample)")
cli_args = parser.parse_args()
ENV_NAME = cli_args.env

NUM_RUNS = cli_args.runs

# ruff: fmt: off
configs = [
    # baseline
    {"label": "vanilla", "pid": False},
    # original PID defaults
    {"label": "pid-default", "pid": True, "kp": 1.0, "ki": 0.1, "kd": 0.05, "gamma": 0.95},
    # reduce D (noise amplifier)
    {"label": "kd=0.01", "pid": True, "kp": 1.0, "ki": 0.1, "kd": 0.01, "gamma": 0.95},
    {"label": "kd=0", "pid": True, "kp": 1.0, "ki": 0.1, "kd": 0.0, "gamma": 0.95},
    # reduce both I and D
    {"label": "ki=0.05,kd=0", "pid": True, "kp": 1.0, "ki": 0.05, "kd": 0.0, "gamma": 0.95},
    {"label": "ki=0.02,kd=0", "pid": True, "kp": 1.0, "ki": 0.02, "kd": 0.0, "gamma": 0.95},
    # push kp above 1
    {"label": "kp=1.1,rest=0", "pid": True, "kp": 1.1, "ki": 0.0, "kd": 0.0, "gamma": 0.95},
    {"label": "kp=1.2,rest=0", "pid": True, "kp": 1.2, "ki": 0.0, "kd": 0.0, "gamma": 0.95},
    # kp>1 + small I
    {"label": "kp=1.1,ki=0.05", "pid": True, "kp": 1.1, "ki": 0.05, "kd": 0.0, "gamma": 0.95},
    {"label": "kp=1.2,ki=0.05", "pid": True, "kp": 1.2, "ki": 0.05, "kd": 0.0, "gamma": 0.95},
    # kp>1 + tiny D
    {"label": "kp=1.1,kd=0.01", "pid": True, "kp": 1.1, "ki": 0.05, "kd": 0.01, "gamma": 0.95},
    # Scheduled PID (SNR and ESS variants)
    {"label": "sched-snr", "pid_schedule": "snr", "kp": 1.0, "ki": 0.05, "kd": 0.0},
    {"label": "sched-ess", "pid_schedule": "ess", "kp": 1.0, "ki": 0.05, "kd": 0.0},
    {"label": "sched-snr-full", "pid_schedule": "snr", "kp": 1.0, "ki": 0.00, "kd": 0.05},
    {"label": "sched-ess-full", "pid_schedule": "ess", "kp": 1.0, "ki": 0.00, "kd": 0.05},
    {"label": "sched-snr-full", "pid_schedule": "snr", "kp": 1.0, "ki": 0.05, "kd": 0.05},
    {"label": "sched-ess-full", "pid_schedule": "ess", "kp": 1.0, "ki": 0.05, "kd": 0.05},
    {"label": "sched-snr-full", "pid_schedule": "snr", "kp": 1.0, "ki": 0.10, "kd": 0.05},
    {"label": "sched-ess-full", "pid_schedule": "ess", "kp": 1.0, "ki": 0.10, "kd": 0.05},
    {"label": "sched-snr-full", "pid_schedule": "snr", "kp": 1.0, "ki": 0.05, "kd": 0.10},
    {"label": "sched-ess-full", "pid_schedule": "ess", "kp": 1.0, "ki": 0.05, "kd": 0.10},
    {"label": "sched-snr-full", "pid_schedule": "snr", "kp": 1.0, "ki": 0.10, "kd": 0.10},
    {"label": "sched-ess-full", "pid_schedule": "ess", "kp": 1.0, "ki": 0.10, "kd": 0.10},
]
# ruff: fmt: on

results = {}
all_histories = {}  # label -> list of history arrays

for cfg in configs:
    label = cfg.pop("label")
    print(f"\n{'=' * 60}")
    print(f"  {label}  |  {cfg}")
    print(f"{'=' * 60}")
    extra = {}
    if cli_args.steps is not None:
        extra["Ndiffuse"] = cli_args.steps
    if cli_args.samples is not None:
        extra["Nsample"] = cli_args.samples
    args = MBDArgs(env=ENV_NAME, not_render=True, **cfg, **extra)
    rews = []
    histories = []
    for r in range(NUM_RUNS):
        t0 = time.time()
        rew, hist = run_diffusion(args)
        elapsed = time.time() - t0
        rews.append(float(rew))
        histories.append(hist)
        print(f"  run {r + 1}: reward={float(rew):.4f}, time={elapsed:.1f}s")
    avg = sum(rews) / len(rews)
    std = float(np.std(rews))
    results[label] = (avg, std, min(rews), max(rews))
    all_histories[label] = histories
    print(f"  => avg={avg:.4f}  std={std:.4f}  min={min(rews):.4f}  max={max(rews):.4f}")
    cfg["label"] = label  # restore

print(f"\n\n{'=' * 80}")
print(f"{'Config':<25} {'Avg':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print(f"{'-' * 61}")
for label, (avg, std, lo, hi) in results.items():
    print(f"{label:<25} {avg:>8.4f} {std:>8.4f} {lo:>8.4f} {hi:>8.4f}")
print(f"{'=' * 80}")

# --- Save to timestamped benchmarks directory ---
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
out_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "benchmarks",
    f"{timestamp}_sweep_pid_{ENV_NAME}",
)
os.makedirs(out_dir, exist_ok=True)

summary_path = os.path.join(out_dir, "summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "avg_reward", "std_reward", "min_reward", "max_reward"])
    for label, (avg, std, lo, hi) in results.items():
        writer.writerow([label, f"{avg:.6f}", f"{std:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

# --- Save reward evolution ---
rows = []
for label, histories in all_histories.items():
    for run_idx, hist in enumerate(histories):
        for step_idx, rew in enumerate(hist):
            rows.append({"config": label, "run": run_idx, "step": step_idx, "reward": float(rew)})
history_df = pd.DataFrame(rows)
history_path = os.path.join(out_dir, "reward_evolution.parquet")
history_df.to_parquet(history_path, index=False)

# --- Plot reward evolution ---
fig, ax = plt.subplots(figsize=(10, 6))
for label in all_histories:
    df_l = history_df[history_df["config"] == label]
    grouped = df_l.groupby("step")["reward"]
    mean = grouped.mean()
    std = grouped.std().fillna(0)
    ax.plot(mean.index, mean.values, label=label)
    ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.15)
ax.set_xlabel("Diffusion step")
ax.set_ylabel("Mean sample reward")
ax.set_title(f"Reward evolution — PID sweep ({ENV_NAME})")
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)
fig.tight_layout()
plot_path = os.path.join(out_dir, "reward_evolution.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)

print(f"\nResults saved to: {out_dir}")
