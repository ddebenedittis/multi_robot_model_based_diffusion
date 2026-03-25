# CLAUDE.md

## Architecture

```
mbd/                - Modified Model-Based Diffusion
src/
└── mrmbd/          - Multi-Robot Model-Based Diffusion (mrmbd) package
    ├── envs/       - Environment dynamics & kinematics
    ├── planners/   - LID/LIDEC planning algorithms
    ├── scripts/    - Visualization & inference entry points
    ├── utils.py    - Cost functions, Augmented Lagrangian, noise schedules
    └── butterworth.py - Low-pass noise filtering
```

### Three Domains

| Domain | Environment class | Planner | Description |
|--------|------------------|---------|-------------|
| Multi-Car | `MultiCar2d` (`envs/multi_car.py`) | `lid_multicar.py` | 4 unicycles, collision avoidance, formation tasks |
| RRPR Manipulator | `RRPRSingleEnv` (`envs/class_manipulator.py`) | `lid_rrpr.py` | 4-DOF arm, end-effector trajectory tracking |
| Crane Pendulum | `CranePendulumEnv` (`envs/class_overhead_crane.py`) | `lid_crane.py` | Cart + pendulum swing-up control |

### Key Patterns

- **Never commit changes**
- Save all outputs in timestamped subdirectories (`yyyymmdd_hhmmss_shortname`) or either `results/` or `benchmarks/`.
- **JAX-first**: all dynamics/costs use `jax.jit`, `jax.vmap`, `jax.lax.scan` for vectorized computation.
- **Flax `@struct.dataclass`** for state objects (not standard dataclasses).
