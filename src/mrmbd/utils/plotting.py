"""Shared plotting style and helpers for unified figure styling.

Centralizes the academic matplotlib style that was previously duplicated across
the multi-car plotting code, and provides a rotated-triangle marker for drawing
unicycle robots. Adapted from ``hierarchical_optimization_mpc``'s
``disp_het_multi_rob.py`` (``init_matplotlib`` / ``gen_arrow_head_marker``).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.animation import FFMpegWriter

# Single source of truth for per-robot trajectory colors.
ROBOT_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def set_plot_style(usetex: bool = True):
    """Apply the unified academic plotting style to matplotlib's rcParams.

    Args:
        usetex: When True (default), render text with a real LaTeX engine and a
            math preamble (publication quality, requires a working LaTeX install).
            When False, fall back to serif fonts + mathtext, which has no external
            dependency.
    """
    default_cycler = cycler(color=ROBOT_PALETTE + ["#4DBEEE", "#A2142F"]) + cycler(
        "linestyle", ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["CMU Serif", "DejaVu Serif", "Times"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "axes.prop_cycle": default_cycler,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "grid.linestyle": "-.",
            "grid.alpha": 0.5,
            "figure.constrained_layout.use": True,
        }
    )

    if usetex:
        plt.rcParams.update(
            {
                "text.usetex": True,
                "text.latex.preamble": (
                    r"\usepackage[utf8]{inputenc} \usepackage{amsmath} "
                    r"\usepackage{amsfonts}"
                ),
            }
        )
    else:
        plt.rcParams["text.usetex"] = False


def save_animation(ani, path, fps=10, dpi=100, preset="ultrafast"):
    """Save a matplotlib animation to MP4 with fast H.264 encoding.

    Speeds up writing relative to ``ani.save(path, fps=..., dpi=150)`` by
    (1) configuring an explicit :class:`FFMpegWriter` with a fast x264 preset and
    all CPU threads, and (2) defaulting to a lower ``dpi`` (the pixel count drives
    both per-frame rasterization and encoding cost).

    Args:
        ani: A :class:`matplotlib.animation.FuncAnimation` (or any Animation).
        path: Output ``.mp4`` path.
        fps: Frames per second.
        dpi: Output resolution; 100 is plenty for results/embedded videos.
        preset: x264 speed/size tradeoff (``ultrafast`` is the fastest encode).
    """
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        extra_args=[
            "-preset",
            preset,
            "-threads",
            "0",
            "-pix_fmt",
            "yuv420p",
            # yuv420p needs even dimensions; round down odd width/height to nearest even.
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ],
    )
    ani.save(path, writer=writer, dpi=dpi)


def _draw_obstacle_penalty_zones(ax, env):
    """Draw static obstacles and their soft-penalty buffer zones."""
    buffer_min, buffer_max = 0.2, 0.5
    for x_c, y_c, w, h in env.static_obstacles:
        ax.add_patch(
            plt.Rectangle(
                (x_c - w / 2, y_c - h / 2),
                w,
                h,
                linewidth=1.0,
                edgecolor="black",
                facecolor="#d3d3d3",
                zorder=1,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
                w + 2 * buffer_max,
                h + 2 * buffer_max,
                linewidth=0.8,
                edgecolor="none",
                facecolor="#a6bddb",
                alpha=0.25,
                zorder=1,
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
                w + 2 * buffer_min,
                h + 2 * buffer_min,
                linewidth=0.8,
                edgecolor="none",
                facecolor="#3690c0",
                alpha=0.35,
                zorder=2,
            )
        )


def plot_diffusion_cloud(
    out_dir,
    env,
    palette=ROBOT_PALETTE,
    max_samples=80,
    fps=10,
    save_video=True,
    xlim=(-5, 5),
    ylim=(-5, 5),
):
    """Render the cloud of multi-car diffusion sample trajectories.

    Reads ``global_Yi_list.npz`` from ``out_dir`` (written by the multi-car
    planner when ``save_video`` is enabled) and saves, into the same directory:

    - ``global_diffusion_initial.png`` / ``global_diffusion_final.png`` — static
      snapshots of the first and last reverse-diffusion steps;
    - ``global_diffusion_video.mp4`` — the full evolution (only when ``save_video``).

    Each frame overlays up to ``max_samples`` semi-transparent stochastic sample
    trajectories per robot with the bold optimized (denoised) trajectory.

    Args:
        out_dir: Directory containing ``global_Yi_list.npz``; outputs go here too.
        env: A ``MultiCar2d`` instance (provides ``n``, ``xg``, ``static_obstacles``).
        palette: Per-robot colors.
        max_samples: Max number of sample trajectories drawn per robot per frame.
        fps: Frames per second for the video.
        save_video: When False, only the static snapshots are written.
        xlim, ylim: Axis limits for the video frames.
    """
    import os

    import matplotlib.animation as animation
    from matplotlib.lines import Line2D

    npz_path = os.path.join(out_dir, "global_Yi_list.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"No global_Yi_list.npz found in {out_dir!r}. "
            "Run the planner with --save_video to produce the diffusion-cloud data."
        )

    data = np.load(npz_path)
    trajectories_all = data["trajectories_samples"]  # (Nsteps, Nsample, T+1, n, 2)
    trajectories_denoised = data["trajectories_denoised"]  # (Nsteps, T+1, n, 2)
    n = env.n

    # === Static snapshots (initial and final reverse-diffusion step) ===
    for tag, idx in {"initial": 0, "final": len(trajectories_all) - 1}.items():
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect("equal")
        ax.grid(True, linestyle="-", color="k", linewidth=0.6, alpha=0.7)
        _draw_obstacle_penalty_zones(ax, env)

        samples = trajectories_all[idx]
        for i in range(n):
            for s in range(min(max_samples, samples.shape[0])):
                ax.plot(samples[s][:, i, 0], samples[s][:, i, 1], alpha=0.07, color=palette[i])

        traj_opt = trajectories_denoised[idx]
        for i in range(n):
            ax.plot(
                env.xg[i, 0],
                env.xg[i, 1],
                marker="s",
                color=palette[i],
                markersize=5.5,
                markeredgewidth=0.8,
                zorder=5,
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
        plt.savefig(os.path.join(out_dir, f"global_diffusion_{tag}.png"), dpi=300)
        plt.close(fig)
    print("Static images saved: global_diffusion_initial.png, global_diffusion_final.png")

    if not save_video:
        return

    # === Animated evolution across all reverse-diffusion steps ===
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(frame):
        ax.clear()
        ax.set_title(f"Reverse Diffusion Step {frame}")
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="-", color="k", linewidth=0.6, alpha=0.7)
        _draw_obstacle_penalty_zones(ax, env)

        samples = trajectories_all[frame]
        for i in range(n):
            for s in range(min(max_samples, samples.shape[0])):
                ax.plot(samples[s][:, i, 0], samples[s][:, i, 1], alpha=0.07, color=palette[i])

        traj_opt = trajectories_denoised[frame]
        for i in range(n):
            ax.plot(traj_opt[:, i, 0], traj_opt[:, i, 1], color=palette[i], linewidth=2)
            ax.plot(traj_opt[0, i, 0], traj_opt[0, i, 1], "o", color=palette[i], markersize=4)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(trajectories_all), interval=150)
    output_path = os.path.join(out_dir, "global_diffusion_video.mp4")
    save_animation(ani, output_path, fps=fps)
    plt.close(fig)
    print(f"Video saved: {output_path}")


def gen_arrow_head_marker(rot):
    """Generate a rotated triangular marker to plot a unicycle robot.

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction

    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """

    arr = np.array([[0.1, 0.3], [0.1, -0.3], [1, 0], [0.1, 0.3]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [
        mpl.path.Path.MOVETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.CLOSEPOLY,
    ]
    arrow_head_marker = mpl.path.Path(arr, codes)

    return arrow_head_marker, scale
