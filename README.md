# Multi-Robot Model-Based Diffusion (mrmbd)

This package extends the [Model-Based Diffusion](https://arxiv.org/abs/2305.12577) (MBD) framework to multi-robot systems.

It implements:
- **LID** (Local Iterative Diffusion): two-phase method with global reverse diffusion followed by local iterative refinement on sliding windows.
- **LIDEC** (LID + Equality Constraints): extends LID with Augmented Lagrangian optimization for hard equality constraints.

These are applied to three domains: multi-car (4 unicycles), RRPR manipulator (4-DOF arm), and crane pendulum (cart + pendulum swing-up).

## Installation

```bash
# Install the original MBD from the submodule (for running original experiments)
pip install -e ./mbd

# Install mrmbd
pip install -e .
```

## Reproducing Results

### Multi-Car: LID (two-phase, no constraints)
```bash
python -m mrmbd.planners.lid_multicar --n_robots 4 [--ECD] [--obstacles_enabled] [--formation_shift] [--penalize_backward] [--cosine] [--filter]
```
Where:
- `--ECD`: enable equality constraints (LIDEC variant)
- `--obstacles_enabled`: add static obstacles to the environment
- `--formation_shift`: perform a formation shift task instead of point-to-point
- `--penalize_backward`: add a penalty for backward motion in the cost function
- `--cosine`: use a cosine noise schedule instead of linear
- `--filter`: apply a Butterworth filter to the noise samples

### RRPR Manipulator
```bash
python -m mrmbd.planners.lid_rrpr
```

### Crane Pendulum
```bash
python -m mrmbd.planners.lid_crane
```

### Visualization / Post-processing
```bash
python -m mrmbd.scripts.graphic           # multi-car analysis plots
python -m mrmbd.scripts.graphic_crane     # crane diffusion animation
python -m mrmbd.scripts.graphic_man       # RRPR 3D EE trajectory animation
python -m mrmbd.scripts.cosine            # noise schedule comparison plots
```
