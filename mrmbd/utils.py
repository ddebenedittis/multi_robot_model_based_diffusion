import dataclasses
import os
from datetime import datetime

import jax
import yaml
from jax import numpy as jnp


def create_experiment_dir(experiment_name: str, args, variant_tags: list[str] | None = None) -> str:
    """Create a timestamped experiment directory and save parameters.

    Args:
        experiment_name: Base name for the experiment (e.g. "crane", "rrpr", "multicar").
        args: Dataclass or flax struct.dataclass with experiment parameters.
        variant_tags: Optional list of tags appended to the directory name.

    Returns:
        Path to the created experiment directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_name = f"{timestamp}-{experiment_name}"
    if variant_tags:
        dir_name += "-" + "-".join(variant_tags)

    dir_path = os.path.join("results", dir_name)
    os.makedirs(dir_path, exist_ok=True)

    # Serialize args to dict
    try:
        params = dataclasses.asdict(args)
    except TypeError:
        # Fallback for flax struct.dataclass: iterate over fields manually
        params = {}
        for field in dataclasses.fields(args):
            val = getattr(args, field.name)
            # Convert JAX/numpy arrays to Python scalars
            if hasattr(val, "item"):
                val = val.item()
            params[field.name] = val

    with open(os.path.join(dir_path, "params.yaml"), "w") as f:
        yaml.dump(params, f, default_flow_style=False)

    # Create/update symlink results/latest-{experiment_name}
    symlink_path = os.path.join("results", f"latest-{experiment_name}")
    if os.path.islink(symlink_path):
        os.remove(symlink_path)
    elif os.path.exists(symlink_path):
        os.remove(symlink_path)
    os.symlink(dir_name, symlink_path)

    return dir_path


def rollout_multi_us(step_env, state, us):
    """Rollout for multi-robot systems.

    Args:
        step_env: environment step function (env.step)
        state: initial state
        us: controls, shape (H, n, 2)

    Returns:
        rews: rewards at each timestep, shape (H,)
        pipeline_states: full state trajectories, shape (H, n, 3)
    """

    def step(state, u_t):
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state)

    _, (rews, pipeline_states) = jax.lax.scan(step, state, us)
    return rews, pipeline_states


# === Cost functions for ECD optimization ===


@jax.jit
def cost_fn(U_seq):
    """Quadratic cost on control sequences."""
    return jnp.sum(U_seq**2, axis=(1, 2, 3))


def make_reverse_penalty_cost():
    """Penalize reverse motion (v < 0) with quadratic penalty."""

    def sample_penalty(U_seq):
        v = U_seq[:, :, 1]  # linear velocity
        retro = jnp.minimum(v, 0)
        penalty = 50 * jnp.mean(retro**2)
        return penalty

    return jax.vmap(sample_penalty)


def make_goal_tracking_cost(xg, x0):
    """Normalized goal tracking cost for batches of local trajectories.

    Args:
        xg: robot goals, shape (n, 3)
        x0: initial positions, shape (n, 3)

    Returns:
        goal_cost_fn: function (trajs_window) -> (N,)
    """
    goal_pos = xg[:, :2]
    start_pos = x0[:, :2]
    goal_dists = jnp.linalg.norm(start_pos - goal_pos, axis=1) + 1e-6

    def sample_cost(traj):
        pos = traj[:, :, :2]
        diff = pos - goal_pos
        dists = jnp.linalg.norm(diff, axis=-1)
        norm_dists = dists / goal_dists
        return jnp.mean(norm_dists)

    return jax.vmap(sample_cost)


def make_log_barrier_collision_cost(obs, n, Ra, epsilon=1e-6):
    """Log-barrier collision cost: positive near collisions, zero far away."""

    def cost_fn(trajs):
        def sample_cost(traj):
            def timestep_cost(state_t):
                pos = state_t[:, :2]
                idx_i, idx_j = jnp.triu_indices(n, k=1)
                diffs = pos[idx_i] - pos[idx_j]
                dists = jnp.linalg.norm(diffs, axis=1)

                dist_safe = jnp.clip(dists - 2 * Ra, a_min=epsilon)
                barrier = -jnp.log(dist_safe) if obs else jnp.maximum(-jnp.log(dist_safe), 0.0)

                return jnp.mean(barrier)

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return cost_fn


def make_formation_cost_fn(x0_all):
    """Formation deformation cost relative to initial inter-robot distances."""
    pos0 = x0_all[:, :2]
    diff0 = pos0[:, None, :] - pos0[None, :, :]
    dists0 = jnp.linalg.norm(diff0, axis=-1)
    mask = jnp.triu(jnp.ones((x0_all.shape[0], x0_all.shape[0]), dtype=bool), k=1)

    def formation_cost(trajs):
        def sample_cost(traj):
            def timestep_cost(state_t):
                pos = state_t[:, :2]
                diff = pos[:, None, :] - pos[None, :, :]
                dists = jnp.linalg.norm(diff, axis=-1)
                return jnp.sum(((dists - dists0) ** 2) * mask)

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return formation_cost


def make_residual_fn(penalize, state_init, env, Nsample):
    """Creates a residual function that measures final position error from goal."""

    @jax.jit
    def residual_fn(pipeline_states):
        s0 = state_init.pipeline_state
        s0_batched = jnp.repeat(s0[None, ...], Nsample, axis=0)
        trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1)
        x_Ts = trajs[:, -1, :, :2]
        theta_Ts = trajs[:, -1, :, 2]
        goal_error = x_Ts - env.xg[:, :2]
        if penalize:
            # Angular error (correct angular difference)
            theta_g = env.xg[:, 2]
            theta_diff = jnp.arctan2(jnp.sin(theta_Ts - theta_g), jnp.cos(theta_Ts - theta_g))
            residual = jnp.concatenate([goal_error, theta_diff[:, :, None]], axis=-1)
        else:
            residual = x_Ts - env.xg[:, :2]
        return residual

    return residual_fn


def make_static_obstacle_cost(static_obs, robot_radius, epsilon=1e-3, margin=0.05, scale=1.0):
    """Log-barrier + 1/x cost for proximity to static obstacles."""

    def cost_fn(trajs):
        def sample_cost(traj):
            def timestep_cost(state_t):
                x = state_t[:, 0]
                y = state_t[:, 1]
                cost = 0.0

                for x_c, y_c, w, h in static_obs:
                    obs_x_min = x_c - w / 2
                    obs_x_max = x_c + w / 2
                    obs_y_min = y_c - h / 2
                    obs_y_max = y_c + h / 2

                    dx = jnp.maximum(jnp.maximum(obs_x_min - x, 0.0), x - obs_x_max)
                    dy = jnp.maximum(jnp.maximum(obs_y_min - y, 0.0), y - obs_y_max)
                    dist = jnp.sqrt(dx**2 + dy**2 + epsilon)
                    x_clipped = jnp.minimum(dist / 0.5, 1.0)
                    barrier = jnp.log(x_clipped) / jnp.log(2 * robot_radius / 0.5)
                    penalty = 100 * barrier
                    cost += jnp.mean(penalty)

                return cost

            return jnp.mean(jax.vmap(timestep_cost)(traj))

        return jax.vmap(sample_cost)(trajs)

    return cost_fn


def make_orient_final_cost_fn(xg, w_theta=1.0, decay=10.0):
    """Final orientation cost weighted by proximity to goal."""
    goal_pos = xg[:, :2]
    theta_g = xg[:, 2]

    def sample_cost(traj):
        pos_T = traj[-1, :, :2]
        theta_T = traj[-1, :, 2]

        dists = jnp.linalg.norm(pos_T - goal_pos, axis=-1)
        weights = jnp.exp(-decay * dists)
        theta_diff = jnp.arctan2(jnp.sin(theta_T - theta_g), jnp.cos(theta_T - theta_g))
        penalties = weights * theta_diff**2
        return w_theta * jnp.mean(penalties)

    return jax.vmap(sample_cost)


def make_lagrangian_fn(state_init, env, Nsample):
    """Lagrangian function for equality-constrained optimization (LIDEC)."""
    residual_fn = make_residual_fn(env.penalize_backward, state_init, env, Nsample)
    log_barrier_fn = make_log_barrier_collision_cost(
        env.obstacles_enabled, env.n, env.Ra, epsilon=1e-3
    )
    goal_cost_fn = make_goal_tracking_cost(env.xg, env.x0)
    formation_cost_fn = make_formation_cost_fn(env.x0) if env.formation_shift else lambda x: 0.0
    obstacle_cost_fn = (
        make_static_obstacle_cost(env.static_obstacles, env.Ra)
        if env.obstacles_enabled
        else lambda x: 0.0
    )
    reverse_penalty_fn = make_reverse_penalty_cost() if env.penalize_backward else lambda x: 0.0
    orient_cost_fn = make_orient_final_cost_fn(env.xg) if env.penalize_backward else lambda x: 0.0

    @jax.jit
    def lagrangian(Y0s_windows, Y0s, pipeline_states, pipeline_states_window, lambda_goal, mu_k):
        s0 = state_init.pipeline_state
        s0_batched = jnp.repeat(s0[None, ...], Y0s.shape[0], axis=0)
        trajs = jnp.concatenate([s0_batched[:, None, :, :], pipeline_states], axis=1)

        control_cost_global = cost_fn(Y0s)
        control_cost_local = cost_fn(Y0s_windows)

        reverse_penalty_global = reverse_penalty_fn(Y0s)

        barrier_cost_global = log_barrier_fn(trajs)
        barrier_cost_local = log_barrier_fn(pipeline_states_window)

        goal_cost_global = goal_cost_fn(trajs)
        goal_cost_local = goal_cost_fn(pipeline_states_window)

        formation_cost_global = formation_cost_fn(trajs)
        formation_cost_local = formation_cost_fn(pipeline_states_window)

        obstacle_cost_global = obstacle_cost_fn(trajs)
        obstacle_cost_local = obstacle_cost_fn(pipeline_states_window)

        orient_cost_global = orient_cost_fn(trajs)

        h = residual_fn(pipeline_states)
        h_flat = h.reshape((Y0s.shape[0], -1))

        L_cost_local = (
            control_cost_local
            + 30 * barrier_cost_local
            + 20 * goal_cost_local
            + 15 * formation_cost_local
            + 10 * obstacle_cost_local
        )
        L_cost_global = (
            0.5 * control_cost_global
            + 25 * barrier_cost_global
            + 30 * goal_cost_global
            + 15 * formation_cost_global
            + 10 * obstacle_cost_global
            + 30 * orient_cost_global
            + 10 * reverse_penalty_global
        )

        L_constraint = jnp.dot(h_flat, lambda_goal) + 0.5 * mu_k * jnp.sum(h_flat**2, axis=1)
        L_tot = L_cost_global + L_constraint

        return (
            L_cost_local,
            L_constraint,
            L_tot,
            control_cost_global,
            barrier_cost_global,
            goal_cost_global,
            h_flat,
            obstacle_cost_global,
            reverse_penalty_global,
            orient_cost_global,
        )

    return lagrangian


# === Noise schedule functions ===


def cosine_beta_schedule(T, s=0.008):
    """Cosine schedule for betas, as in Improved DDPM."""
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas
    return jnp.clip(betas, 1e-5, 0.999)


def cosine_beta_schedule_scaled(T, beta0, betaT, s=0.008):
    """Cosine schedule with linear scaling to match desired beta0 and betaT."""
    t = jnp.arange(T + 1, dtype=jnp.float32)
    f_t = jnp.cos(((t / T + s) / (1 + s)) * jnp.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    alphas = alphas_bar[1:] / alphas_bar[:-1]
    betas = 1 - alphas

    beta_min, beta_max = betas.min(), betas.max()
    betas_scaled = (betas - beta_min) / (beta_max - beta_min)
    betas_scaled = betas_scaled * (betaT - beta0) + beta0

    return betas_scaled
