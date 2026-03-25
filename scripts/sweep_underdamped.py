"""Sweep underdamped Langevin dynamics parameters for MBD on a given environment."""

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
import numpy as np

print(f"JAX backend: {jax.default_backend()}")

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="ant", help="Environment name (ant, halfcheetah, ...)")
parser.add_argument("--runs", type=int, default=3, help="Number of runs per config")
parser.add_argument("--steps", type=int, default=None, help="Number of diffusion steps (Ndiffuse)")
parser.add_argument("--samples", type=int, default=None, help="Number of samples (Nsample)")
cli_args = parser.parse_args()
ENV_NAME = cli_args.env

from mbd.planners.mbd_planner import Args as MBDArgs
from mbd.planners.mbd_planner import run_diffusion

NUM_RUNS = cli_args.runs

# ruff: fmt: off
configs = [
    # baseline (no underdamped)
    {"label": "vanilla", "underdamped": False},
    # defaults
    {
        "label": "uld-default",
        "underdamped": True,
        "friction": 0.5,
        "mass": 1.0,
        "velocity_clip": 2.0,
    },
    # vary friction
    {
        "label": "friction=0.1",
        "underdamped": True,
        "friction": 0.1,
        "mass": 1.0,
        "velocity_clip": 2.0,
    },
    {
        "label": "friction=0.3",
        "underdamped": True,
        "friction": 0.3,
        "mass": 1.0,
        "velocity_clip": 2.0,
    },
    {
        "label": "friction=0.7",
        "underdamped": True,
        "friction": 0.7,
        "mass": 1.0,
        "velocity_clip": 2.0,
    },
    {
        "label": "friction=0.9",
        "underdamped": True,
        "friction": 0.9,
        "mass": 1.0,
        "velocity_clip": 2.0,
    },
    # vary mass
    {"label": "mass=0.5", "underdamped": True, "friction": 0.5, "mass": 0.5, "velocity_clip": 2.0},
    {"label": "mass=2.0", "underdamped": True, "friction": 0.5, "mass": 2.0, "velocity_clip": 2.0},
    {"label": "mass=5.0", "underdamped": True, "friction": 0.5, "mass": 5.0, "velocity_clip": 2.0},
    # vary velocity clip
    {"label": "vclip=1.0", "underdamped": True, "friction": 0.5, "mass": 1.0, "velocity_clip": 1.0},
    {"label": "vclip=5.0", "underdamped": True, "friction": 0.5, "mass": 1.0, "velocity_clip": 5.0},
    {
        "label": "vclip=10.0",
        "underdamped": True,
        "friction": 0.5,
        "mass": 1.0,
        "velocity_clip": 10.0,
    },
    # low friction + low mass (more momentum)
    {
        "label": "fric=0.1,mass=0.5",
        "underdamped": True,
        "friction": 0.1,
        "mass": 0.5,
        "velocity_clip": 2.0,
    },
    {
        "label": "fric=0.3,mass=0.5",
        "underdamped": True,
        "friction": 0.3,
        "mass": 0.5,
        "velocity_clip": 2.0,
    },
    # low friction + higher clip
    {
        "label": "fric=0.1,vclip=5.0",
        "underdamped": True,
        "friction": 0.1,
        "mass": 1.0,
        "velocity_clip": 5.0,
    },
    {
        "label": "fric=0.3,vclip=5.0",
        "underdamped": True,
        "friction": 0.3,
        "mass": 1.0,
        "velocity_clip": 5.0,
    },
]
# ruff: fmt: on

results = {}

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
    for r in range(NUM_RUNS):
        t0 = time.time()
        rew, _hist = run_diffusion(args)
        elapsed = time.time() - t0
        rews.append(float(rew))
        print(f"  run {r + 1}: reward={float(rew):.4f}, time={elapsed:.1f}s")
    avg = sum(rews) / len(rews)
    std = float(np.std(rews))
    results[label] = (avg, std, min(rews), max(rews))
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
    f"{timestamp}_sweep_underdamped_{ENV_NAME}",
)
os.makedirs(out_dir, exist_ok=True)

summary_path = os.path.join(out_dir, "summary.csv")
with open(summary_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["config", "avg_reward", "std_reward", "min_reward", "max_reward"])
    for label, (avg, std, lo, hi) in results.items():
        writer.writerow([label, f"{avg:.6f}", f"{std:.6f}", f"{lo:.6f}", f"{hi:.6f}"])

print(f"\nResults saved to: {out_dir}")
