# Multi-Robot Model-Based Diffusion (mrmbd)

This package extends the [Model-Based Diffusion](https://arxiv.org/abs/2407.01573) (MBD) framework to multi-robot trajectory optimization. It builds on the [MBD codebase](mbd/README.md) (included as a submodule) and introduces two new algorithms:

- **LID** (Local Iterative Diffusion): a two-phase method that first runs global reverse diffusion from noise to an initial trajectory, then refines it locally on overlapping sliding windows.
- **LIDEC** (LID + Equality Constraints): extends LID with Augmented Lagrangian optimization for hard equality constraints (e.g., exact goal reaching).

## Domains

| Domain | Environment | Planner | Description |
|--------|-------------|---------|-------------|
| Multi-Car | `MultiCar2d` | `lid_multicar.py` | 4 unicycle robots with inter-robot collision avoidance, static obstacles, and formation tasks |
| RRPR Manipulator | `RRPRSingleEnv` | `lid_rrpr.py` | 4-DOF revolute-revolute-prismatic-revolute arm for end-effector trajectory tracking |
| Crane Pendulum | `CranePendulumEnv` | `lid_crane.py` | Overhead crane (cart on rail + suspended pendulum) swing-up control |

## Installation

Clone the repository with submodules:
```bash
git clone --recurse-submodules <repo-url>
cd mrmbd
```

This project uses [uv](https://docs.astral.sh/uv/):
```bash
uv sync --extra cuda12
uv pip install -e ./mbd
source .venv/bin/activate
```

For CUDA support, use `uv sync --extra cuda12` instead.

## Usage

### Multi-Car

```bash
python -m mrmbd.planners.lid_multicar --n_robots 4 [--ECD] [--obstacles_enabled] [--formation_shift] [--penalize_backward] [--cosine] [--filter]
```

| Flag | Description |
|------|-------------|
| `--ECD` | Enable equality constraints (LIDEC variant) |
| `--obstacles_enabled` | Add static obstacles |
| `--formation_shift` | Formation shift task instead of point-to-point |
| `--penalize_backward` | Penalize backward motion in the cost function |
| `--cosine` | Cosine noise schedule (default: linear) |
| `--filter` | Butterworth low-pass filter on noise samples |

### RRPR Manipulator

```bash
python -m mrmbd.planners.lid_rrpr
```

### Crane Pendulum

```bash
python -m mrmbd.planners.lid_crane
```

### Visualization

```bash
python -m mrmbd.scripts.graphic           # multi-car analysis plots
python -m mrmbd.scripts.graphic_crane     # crane pendulum animation
python -m mrmbd.scripts.graphic_man       # RRPR 3D end-effector animation
python -m mrmbd.scripts.cosine            # noise schedule comparison plots
```

### Benchmarking (base MBD environments)

```bash
python scripts/benchmark.py               # benchmark MBD + path integral planners
python scripts/sweep_pid.py               # PID gain sweep for MBD tuning
python scripts/sweep_underdamped.py        # underdamped parameter sweep
```

Results are saved to timestamped subdirectories under `benchmarks/`.

## Project Structure

```
mrmbd/
├── src/mrmbd/              # Main package
│   ├── envs/               # Environment dynamics & kinematics
│   ├── planners/           # LID/LIDEC planning algorithms
│   ├── scripts/            # Visualization & inference entry points
│   ├── utils.py            # Cost functions, Augmented Lagrangian, noise schedules
│   └── butterworth.py      # Low-pass noise filtering
├── scripts/                # Benchmarking & parameter sweep scripts
├── mbd/                    # MBD submodule (LeCAR-Lab/model-based-diffusion)
├── results/                # Experiment outputs (timestamped)
└── benchmarks/             # Benchmark results (timestamped)
```

## Development

Install pre-commit hooks:
```bash
sudo apt install pre-commit  # if not already installed
pre-commit install
```
