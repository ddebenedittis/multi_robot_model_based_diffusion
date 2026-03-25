"""Benchmark all planners on a given environment.

Saves per-step reward evolution, summary statistics, and plots
into a timestamped directory under benchmarks/.

Run from the repo root:
    ~/.venv/bin/python benchmark.py              # defaults to ant
    ~/.venv/bin/python benchmark.py --env halfcheetah
    ~/.venv/bin/python benchmark.py --env ant --runs 5
"""

import argparse
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys

mbd_inner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mbd")
sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(os.path.dirname(__file__))]
sys.path.insert(0, mbd_inner)

from datetime import datetime

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}\n")

from mbd.planners.mbd_planner import Args as MBDArgs
from mbd.planners.mbd_planner import run_diffusion
from mbd.planners.path_integral import Args as PIArgs
from mbd.planners.path_integral import run_path_integral


def bench_mbd(name, env_name, num_runs, **extra_args):
    print("=" * 60)
    print(name)
    print("=" * 60)
    args = MBDArgs(env=env_name, not_render=True, **extra_args)
    runs = []
    for r in range(num_runs):
        t0 = time.time()
        rew, rew_history = run_diffusion(args)
        elapsed = time.time() - t0
        runs.append({"final_reward": float(rew), "time": elapsed, "history": rew_history})
        print(f"  run {r + 1}: reward={float(rew):.4f}, time={elapsed:.1f}s")
    avg_rew = np.mean([r["final_reward"] for r in runs])
    avg_time = np.mean([r["time"] for r in runs])
    print(f"  => avg reward={avg_rew:.4f}, avg time={avg_time:.1f}s\n")
    return runs


def bench_pi(name, env_name, method, num_runs, **extra_overrides):
    print("=" * 60)
    print(name)
    print("=" * 60)
    pi_kwargs = dict(env=env_name, update_method=method)
    if "Nsample" in extra_overrides:
        pi_kwargs["Nsample"] = extra_overrides["Nsample"]
    if "disable_recommended_params" in extra_overrides:
        pi_kwargs["disable_recommended_params"] = extra_overrides["disable_recommended_params"]
    if "Ndiffuse" in extra_overrides:
        pi_kwargs["Nrefine"] = extra_overrides["Ndiffuse"]  # path integral uses Nrefine
    args = PIArgs(**pi_kwargs)
    runs = []
    for r in range(num_runs):
        t0 = time.time()
        rew, rew_history = run_path_integral(args)
        elapsed = time.time() - t0
        runs.append({"final_reward": float(rew), "time": elapsed, "history": rew_history})
        print(f"  run {r + 1}: reward={float(rew):.4f}, time={elapsed:.1f}s")
    avg_rew = np.mean([r["final_reward"] for r in runs])
    avg_time = np.mean([r["time"] for r in runs])
    print(f"  => avg reward={avg_rew:.4f}, avg time={avg_time:.1f}s\n")
    return runs


def build_history_df(all_results):
    """Build a DataFrame with one row per (planner, run, step)."""
    rows = []
    for name, runs in all_results.items():
        for run_idx, run in enumerate(runs):
            for step_idx, rew in enumerate(run["history"]):
                rows.append(
                    {
                        "planner": name,
                        "run": run_idx,
                        "step": step_idx,
                        "reward": rew,
                    }
                )
    return pd.DataFrame(rows)


def build_summary_df(all_results):
    """Build a summary DataFrame with one row per planner."""
    rows = []
    for name, runs in all_results.items():
        finals = [r["final_reward"] for r in runs]
        times = [r["time"] for r in runs]
        rows.append(
            {
                "planner": name,
                "avg_reward": np.mean(finals),
                "std_reward": np.std(finals),
                "min_reward": np.min(finals),
                "max_reward": np.max(finals),
                "avg_time_s": np.mean(times),
                "std_time_s": np.std(times),
                "num_runs": len(runs),
            }
        )
    return pd.DataFrame(rows)


def plot_reward_evolution(history_df, out_path):
    """Plot mean reward +/- std across runs for each planner."""
    fig, ax = plt.subplots(figsize=(10, 6))
    planners = history_df["planner"].unique()
    for name in planners:
        df_p = history_df[history_df["planner"] == name]
        grouped = df_p.groupby("step")["reward"]
        mean = grouped.mean()
        std = grouped.std().fillna(0)
        ax.plot(mean.index, mean.values, label=name)
        ax.fill_between(mean.index, mean.values - std.values, mean.values + std.values, alpha=0.15)
    ax.set_xlabel("Diffusion step")
    ax.set_ylabel("Mean sample reward")
    ax.set_title("Reward evolution during reverse diffusion")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_final_rewards(summary_df, all_results, out_path):
    """Bar chart of final rewards with individual run dots."""
    fig, ax = plt.subplots(figsize=(8, 5))
    planners = summary_df["planner"].values
    x = np.arange(len(planners))
    ax.bar(
        x,
        summary_df["avg_reward"],
        yerr=summary_df["std_reward"],
        capsize=4,
        color="steelblue",
        alpha=0.7,
        zorder=2,
    )
    # scatter individual runs
    for i, name in enumerate(planners):
        finals = [r["final_reward"] for r in all_results[name]]
        ax.scatter([i] * len(finals), finals, color="black", s=20, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(planners, rotation=30, ha="right")
    ax.set_ylabel("Final reward")
    ax.set_title("Final reward comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_timing(summary_df, out_path):
    """Bar chart of average wall-clock time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    planners = summary_df["planner"].values
    x = np.arange(len(planners))
    ax.bar(
        x,
        summary_df["avg_time_s"],
        yerr=summary_df["std_time_s"],
        capsize=4,
        color="coral",
        alpha=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(planners, rotation=30, ha="right")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Runtime comparison")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark MBD planners")
    parser.add_argument(
        "--env", default="ant", help="Environment name (ant, halfcheetah, hopper, walker2d, ...)"
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs per config")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of diffusion steps (Ndiffuse)"
    )
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (Nsample)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test with tiny Nsample/Ndiffuse (1 run, no warmup)",
    )
    cli = parser.parse_args()
    env = cli.env
    num_runs = cli.runs

    dry_overrides = {}
    if cli.dry_run:
        num_runs = 1
        dry_overrides = dict(Nsample=64, Ndiffuse=10, disable_recommended_params=True)
        print("*** DRY RUN: Nsample=64, Ndiffuse=10, 1 run, no warmup ***\n")
    if cli.steps is not None:
        dry_overrides["Ndiffuse"] = cli.steps
    if cli.samples is not None:
        dry_overrides["Nsample"] = cli.samples

    # --- Run all planners ---
    all_results = {}
    all_results["MBD"] = bench_mbd("MBD", env, num_runs, **dry_overrides)
    all_results["MBD+PID"] = bench_mbd("MBD+PID", env, num_runs, pid=True, **dry_overrides)
    all_results["MBD+PID(sched)"] = bench_mbd(
        "MBD+PID(sched)", env, num_runs, pid_schedule="ess", kd=0.0, **dry_overrides
    )
    all_results["MBD+Underdamped"] = bench_mbd(
        "MBD+Underdamped", env, num_runs, underdamped=True, **dry_overrides
    )
    all_results["MPPI"] = bench_pi("MPPI", env, "mppi", num_runs, **dry_overrides)
    all_results["CMA-ES"] = bench_pi("CMA-ES", env, "cma-es", num_runs, **dry_overrides)
    all_results["CEM"] = bench_pi("CEM", env, "cem", num_runs, **dry_overrides)

    # --- Build DataFrames ---
    history_df = build_history_df(all_results)
    summary_df = build_summary_df(all_results)

    # --- Create output directory ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("benchmarks", f"{timestamp}_{env}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Save data ---
    history_path = os.path.join(out_dir, "reward_evolution.parquet")
    summary_path = os.path.join(out_dir, "summary.csv")
    history_df.to_parquet(history_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Saved {history_path}")
    print(f"  Saved {summary_path}")

    # --- Generate plots ---
    print()
    plot_reward_evolution(history_df, os.path.join(out_dir, "reward_evolution.png"))
    plot_final_rewards(summary_df, all_results, os.path.join(out_dir, "final_rewards.png"))
    plot_timing(summary_df, os.path.join(out_dir, "timing.png"))

    # --- Print summary ---
    print(f"\n{'=' * 70}")
    print(summary_df.to_string(index=False))
    print(f"{'=' * 70}")
    print(f"\nAll results saved to: {out_dir}")
