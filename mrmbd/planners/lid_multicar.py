import functools
import jax
from jax import numpy as jnp

import os
from mrmbd.utils import rollout_multi_us, make_lagrangian_fn, make_residual_fn
from mrmbd.utils import cosine_beta_schedule, cosine_beta_schedule_scaled
from mrmbd.envs.multi_car import check_inter_robot_collisions, Args, check_collision_static
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mrmbd.envs import MultiCar2d

import tyro
import numpy as np
import time
from mrmbd.butterworth import butterworth_filter_numpy, ar1_noise_numpy
from mrmbd.butterworth import get_butterworth_coeffs


# === Plot actions for all robots (4 subplots) ===
def plot_all_robot_actions(U_opt, dt=0.1, path="results/multicar_iterative"):
    """
    Creates a figure with 4 subplots (one per robot) showing omega and v over time.
    """
    n = U_opt.shape[1]
    time = np.arange(U_opt.shape[0]) * dt
    os.makedirs(path, exist_ok=True)

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()

    for i in range(n):
        if i >= 4:  # show only the first 4 robots
            break
        axs[i].plot(time, U_opt[:, i, 0], label="omega [rad/s]", color="tab:orange")
        axs[i].plot(time, U_opt[:, i, 1], label="v [m/s]", color="tab:blue")
        axs[i].set_title(f"Robot {i}")
        axs[i].set_xlabel("Time [s]")
        axs[i].set_ylabel("Action")
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    filename = os.path.join(path, "actions_all_robots.pdf")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def plot_obstacle_layout(env):
    """
    Visualize only the initial robot positions and static obstacles,
    with clean academic style suitable for thesis figures.
    """
    plt.rcParams.update({
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
    })

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.set_aspect('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Environment with static obstacles", pad=6)

    # === Static obstacles ===
    for x_c, y_c, w, h in env.static_obstacles:
        rect = plt.Rectangle(
            (x_c - w / 2, y_c - h / 2), w, h,
            linewidth=1.0, edgecolor='black',
            facecolor='#d3d3d3', alpha=0.8, zorder=1
        )
        ax.add_patch(rect)

    # === Initial robot positions ===
    for i, (x, y, _) in enumerate(env.x0):
        ax.plot(x, y, 'o', color=f"C{i}", markersize=6,
                markeredgecolor='k', markeredgewidth=0.6, zorder=3)
        ax.text(x, y - 0.10, f"R{i}", ha='center', va='top', fontsize=8)

    # === Goal positions ===
    for i, (xg, yg, _) in enumerate(env.xg):
        ax.plot(xg, yg, 's', color=f"C{i}", markersize=5,
                markeredgecolor='k', markeredgewidth=0.6, zorder=3)

    # === Dashed grid ===
    ax.grid(True, linestyle="-", color="k", linewidth=0.6, alpha=0.7)

    # === Legend ===
    ax.legend(["Obstacles"], loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig("results/obstacle_layout.pdf", dpi=300)


def run_diffusion_once(args: Args, env, rollout_us, reset_env_jit):
    """
    First phase of D4ORM: initial global reverse diffusion
    inspired by Algorithm 1 (Model-Based Diffusion) from the D4ORM paper.

    """
    rng = jax.random.PRNGKey(seed=args.seed)

    Nx = env.observation_size
    Nu = env.action_size
    n = env.num_robots

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)
    if args.save_video:
        Yi_list = []
        Y0s_list = []
        trajectories_denoised = []
        trajectories_samples = []

    # Diffusion noise schedule
    if args.cosine:
        betas = cosine_beta_schedule(args.Ndiffuse)
    else:
        betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)

    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1 - alphas_bar)

    #  Start from zero control
    YN = jnp.zeros([args.Hsample, n, Nu])

    # Single diffusion step
    def reverse_once(carry):
        i, rng, Ybar_i = carry
        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        # Sample noisy controls
        rng, rng_eps = jax.random.split(rng)

        if args.filter:
            eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))
            eps_u_np = np.array(eps_u)
            b, a = get_butterworth_coeffs(order=4, fc=2.0, fs=1/env.dt)
            eps_u_filt_np = butterworth_filter_numpy(eps_u_np, b, a)
            eps_u = jnp.array(eps_u_filt_np)
        else:
            eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, n, Nu))

        Y0s = eps_u * sigmas[i] + Ybar_i
        if env.obstacles_enabled == False:
            Y0s = jnp.clip(Y0s, -1.0, 1.0)
        rewss, _ = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
        rews = rewss.mean(axis=(1, 2))

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample

        # Weighted average of samples (Monte Carlo estimate)
        weights = jax.nn.softmax(logp0)

        Ybar = jnp.einsum("s,shij->hij", weights, Y0s)
        # Reverse diffusion step
        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])
        return (i - 1, rng, Ybar_im1), Yi, Y0s

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1, args.Ndiffuse)):
            carry = (i, rng, Yi)
            (i, rng, Yi), Yi_current, Y0s = reverse_once(carry)
            if args.save_video and i % 10 == 0:
                # Save Yi
                Yi_list.append(np.array(Yi_current))  # (H, n, Nu)
                Y0s_list.append(np.array(Y0s))  # (Nsample, H, n, Nu)

                # Rollout Yi
                _, traj_denoised = rollout_us(state_init, Yi_current)
                trajectories_denoised.append(np.array(traj_denoised[..., :2]))  # (T+1, n, 2)

                # Rollout Y0s
                _, traj_samples = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, Y0s)
                trajectories_samples.append(np.array(traj_samples[..., :2]))  # (Nsample, T+1, n, 2)
        if args.save_video:
            return Yi, Yi_list, Y0s_list, trajectories_denoised, trajectories_samples
        else:
            return Yi
    if args.save_video:
        rng_exp, rng = jax.random.split(rng)
        U_0, Yi_list, Y0s_list, trajectories_denoised, trajectories_samples = reverse(YN, rng_exp)
        # Save everything
        np.savez("results/multicar_iterative/global_Yi_list.npz",
                Yi_list=Yi_list,
                Y0s_list=Y0s_list,
                trajectories_denoised=trajectories_denoised,
                trajectories_samples=trajectories_samples)
    else:
        rng_exp, rng = jax.random.split(rng)
        U_0 = reverse(YN, rng_exp)
    state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    rewss_eval, _ = rollout_us(state_init_eval, U_0)
    rews, pipeline_states = rollout_us(state_init_eval, U_0)

    reward_per_robot = rewss_eval.mean(axis=0)
    reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
    print(f"global robots average rewards: {reward_array_str}")
    q_states = jnp.concatenate([state_init_eval.pipeline_state[None], pipeline_states[:-1]], axis=0)
    r_terms_all = []  # accumulate r_terms: shape (n, 6)
    for t in range(U_0.shape[0]):
        q_t = q_states[t]
        u_t = U_0[t]
        _, r_terms = env.get_rewards(q_t, u_t)   # shape (n, 6)
        r_terms_all.append(r_terms)

    return U_0

# Local iterative diffusion optimization
def run_diffusion_local(args: Args, U_init: jnp.ndarray, env, rollout_us, reset_env_jit):
    """
    Second phase of D4ORM: local iterative reverse diffusion optimization.
    Based on Algorithm 2 (Iterative Denoising) from the D4ORM paper.
    """
    rng = jax.random.PRNGKey(seed=args.seed + 123)
    rewards_per_iter = []
    if args.save_video:
        trajectories_denoised_local = []
        trajectories_samples_local = []

    H = args.Hsample
    Nu = env.action_size
    n = env.num_robots

    U = U_init.copy()

    # Local diffusion parameters
    L = 10  # window length
    K = 10  # number of local iterations

    # Local diffusion schedule
    betas = jnp.linspace(args.beta0, args.betaT, 10)

    alphas = 1.0 - betas
    alphas_bar_local = jnp.cumprod(alphas)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)

    if args.penalize_backward:
        lambda_goal = jnp.zeros((n * 3))
    else:
        lambda_goal = jnp.zeros((n * 2))

    def reverse_once_local_ECD(U_w, rng_w, lambda_goal, residual_fn, lagrangian, U_full, t_start, reset_env_jit, rollout_us):

            Nsample = args.Nsample
            N_inner = 30  # number of ECD iterations
            U_curr = U_w
            lambda_curr = lambda_goal
            delta_t = 0

            for i in range(N_inner):

                rng_w, rng_step = jax.random.split(rng_w)
                # Noise annealing: exponentially decaying, then set to 0 in last 2 steps
                sigma_k = args.initial_sigma * jnp.exp(-args.noise_decay * i)
                sigma_k = jnp.where(i >= N_inner - 2, 0.0, sigma_k)
                mu_k = args.mu

                # Sample noisy control trajectories generated with gaussian noise
                eps_u = jax.random.normal(rng_step, (Nsample, L, n, Nu))
                noise = eps_u * sigma_k
                Y0s = U_curr + noise
                # note: clip must stay here
                Y0s = jnp.clip(Y0s, -1.0, 1.0)

                U_fulls = jnp.repeat(U_full[None, ...], Nsample, axis=0)
                U_fulls = U_fulls.at[:, t_start:t_start + L, :, :].set(Y0s)
                t1 = time.time()
                state_init = reset_env_jit(rng_w)
                state_init = reset_env_jit(rng_w)

                rewss, pipeline_states = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                t2 = time.time()
                delta_t += t2 - t1
                pipeline_states_window = pipeline_states[:, t_start:t_start+L]

                # Compute Lagrangian values for each sample
                L_cost, L_constraint, L_vals, control_cost, barrier_cost, goal_cost, h_flat, obstacle_cost_global, orient_cost_global, reverse_penalty_global = lagrangian(Y0s, U_fulls, pipeline_states, pipeline_states_window, lambda_curr, args.mu)

                # Estimate gradient using score function estimator
                grad = jnp.einsum("s,slij->lij", L_vals - L_vals.mean(), noise)
                grad = grad / (Nsample * sigma_k ** 2 + 1e-8)  # gradient direction

                rng_key, rng_noise = jax.random.split(rng_w)
                xi = jax.random.normal(rng_noise, U_curr.shape)
                sigma_langevin = sigma_k * jnp.sqrt(2 * args.alpha)
                noise_term = sigma_langevin * xi

                U_next = U_curr - args.alpha * grad + noise_term
                U_next = jnp.clip(U_next, -1.0, 1.0)

                # Update Lagrange multipliers based on average residual
                h_goal = residual_fn(pipeline_states)
                h_mean = jnp.mean(h_goal, axis=0).reshape(-1)
                lambda_next = lambda_curr + args.alpha * mu_k * h_mean

                U_curr = U_next
                lambda_curr = lambda_next
            return U_curr, lambda_curr

    if args.ECD:
        state_init_for_goal = reset_env_jit(jax.random.PRNGKey(args.seed + 777))
        residual_fn = make_residual_fn(env.penalize_backward, state_init_for_goal, env, args.Nsample)
        lagrangian = make_lagrangian_fn(state_init_for_goal, env, args.Nsample)
        print("Final ECD")

        final_ecd_iters = 8
        for i in range(final_ecd_iters):
            for t_start in range(0, H - L + 1, L // 2):
                t_end = t_start + L
                U_window = U[t_start:t_end]
                rng, rng_step = jax.random.split(rng)
                U_opt_local, lambda_goal = reverse_once_local_ECD(U_window, rng, lambda_goal, residual_fn, lagrangian, U, t_start, reset_env_jit, rollout_us)
                U = U.at[t_start:t_end].set(U_opt_local)

            state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed))
            rewss_eval, _ = rollout_us(state_init_eval, U)
            reward_per_robot = rewss_eval.mean(axis=0)
            rewards_per_iter.append(np.array(reward_per_robot))
            reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
            print(f"[Iteration {i}] robots average rewards: {reward_array_str}")
    else:
            for k in range(K):
                for t_start in range(0, H - L + 1, L // 2):  # sliding overlapping windows
                    t_end = t_start + L
                    U_window = U[t_start:t_end]

                    rng, rng_step = jax.random.split(rng)

                    # Local reverse diffusion inside the window
                    def reverse_once_local(U_w, rng_w):
                        for j in reversed(range(1, L)):
                            eps_u = jax.random.normal(rng_w, (args.Nsample, L, n, Nu))
                            sigma_local = sigmas_local[j]
                            Y0s = eps_u * sigma_local + U_w
                            if env.obstacles_enabled == False:
                                 Y0s = jnp.clip(Y0s, -1.0, 1.0)

                            # Insert modified window into full control sequences
                            U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)
                            U_fulls = U_fulls.at[:, t_start:t_end, :, :].set(Y0s)
                            # Evaluate new rollouts
                            state_init = reset_env_jit(rng_step)
                            rewss, traj_samples = jax.vmap(rollout_us, in_axes=(None, 0))(state_init, U_fulls)
                            rews = rewss.mean(axis=(1, 2))

                            # Compute weighted average of samples
                            logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                            weights = jax.nn.softmax(logp0)
                            U_opt = jnp.einsum("s,slij->lij", weights, Y0s)
                            # Final evaluation
                            U_new = jnp.sqrt(alphas_bar_local[j - 1]) * U_opt

                        return U_new

                    U_opt_local = reverse_once_local(U_window, rng_step)
                    U = U.at[t_start:t_end].set(U_opt_local)

                state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))

                rewss_eval, _ = rollout_us(state_init_eval, U)

                reward_per_robot = rewss_eval.mean(axis=0)
                rewards_per_iter.append(np.array(reward_per_robot))
                reward_array_str = "[" + ", ".join(f"{r:.4f}" for r in reward_per_robot) + "]"
                print(f"[Iteration {k}] robots average rewards: {reward_array_str}")
    if args.save_video:
        np.savez("results/multicar_iterative/local_Yi_list.npz",
                 trajectories_denoised=trajectories_denoised_local,
                 trajectories_samples=trajectories_samples_local)

    return U, rewards_per_iter

def main():
    args = tyro.cli(Args)

    total_start = time.time()

    print("STEP 1: Initial Reverse Diffusion")
    env = MultiCar2d(n=args.n_robots, formation_shift=args.formation_shift, ECD=args.ECD, obstacles_enabled=args.obstacles_enabled, penalize_backward=args.penalize_backward)

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    rollout_us = jax.jit(functools.partial(rollout_multi_us, step_env_jit))

    t1 = time.time()
    U_init = run_diffusion_once(args, env, rollout_us, reset_env_jit)
    t2 = time.time()
    print(f"Initial reverse diffusion time: {t2 - t1:.3f} s")

    print("STEP 2: Iterative Local Optimization")
    t3 = time.time()
    U_opt, rewards_per_iter = run_diffusion_local(args, U_init, env, rollout_us, reset_env_jit)
    t4 = time.time()
    print(f"Local optimization time: {t4 - t3:.3f} s")

    total_end = time.time()
    print(f"TOTAL planning time: {total_end - total_start:.3f} s")

    total_time = total_end - total_start
    freq = 1 / total_time
    print(f"Estimated control frequency: {freq:.2f} Hz")

    print("STEP 3: Rollout with optimized controls")

    state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
    _, traj = rollout_us(state_init, U_opt)
    traj = jnp.concatenate([state_init.pipeline_state[None], traj], axis=0)
    traj = jnp.transpose(traj, (1, 0, 2))

    print("Check collisions during rollout:")
    for t in range(traj.shape[1]):
        if check_inter_robot_collisions(traj[:, t, :], env.Ra):
            print(f"Collision detected at timestep {t}")
        if check_collision_static(traj[:, t, :], env.static_obstacles):
            print(f"Collision with static obstacles detected at timestep {t}")

    # Compute final error
    x_goal = env.xg[:, :2]
    x_final = traj[:, -1, :2]
    errors = jnp.linalg.norm(x_final - x_goal, axis=1)

    print("\nFinal average error per robot:")
    for i, err in enumerate(errors):
        print(f"Robot {i}: {err:.3f} m")
    print(f"Mean distance to goal: {errors.mean():.4f} m\n")

    # Interpolate trajectory
    def interpolate_trajectory_jax(xs: jnp.ndarray, dt_original: float = 0.1, dt_interp: float = 0.01):
        n, T, d = xs.shape
        t_max = (T - 1) * dt_original
        t_interp = jnp.arange(0.0, t_max + dt_interp, dt_interp)

        t_original = jnp.arange(0.0, T * dt_original, dt_original)

        def interp_single_robot(traj):
            def interpolate_one(ti):
                idx = jnp.floor(ti / dt_original).astype(int)
                idx = jnp.clip(idx, 0, T - 2)
                t0 = t_original[idx]
                t1 = t_original[idx + 1]
                x0 = traj[idx]
                x1 = traj[idx + 1]
                alpha = (ti - t0) / (t1 - t0)
                return (1 - alpha) * x0 + alpha * x1

            return jax.vmap(interpolate_one)(t_interp)

        xs_interp = jax.vmap(interp_single_robot)(xs)
        return xs_interp, t_interp


    def plot_and_save_reward_terms(r_terms_all, path, prefix="reward", robot_ids=None, component_names=None):
        """
        Plot and save reward components for each robot over time.

        Args:
            r_terms_all: array shape (n, T+1, 6)
            path: directory to save plots
            prefix: filename prefix
            robot_ids: optional list of robots to plot (default: all)
            component_names: names of the reward components (default: auto-generated)
        """
        os.makedirs(path, exist_ok=True)
        n, T_plus_1, num_terms = r_terms_all.shape
        T = T_plus_1 - 1

        if robot_ids is None:
            robot_ids = list(range(n))
        if component_names is None:
            component_names = [f"term{i}" for i in range(num_terms)]

        for j in range(num_terms):
            plt.figure(figsize=(6, 4))
            for k in robot_ids:
                plt.plot(r_terms_all[k, :, j], label=f"Robot {k}")
            plt.xlabel("Time")
            plt.ylabel(component_names[j])
            plt.title(f"{component_names[j]} vs time")
            plt.grid(True)
            plt.legend()
            filename = f"{prefix}_{component_names[j]}.pdf"
            plt.tight_layout()
            plt.savefig(os.path.join(path, filename))
            plt.close()

    if not args.not_render:
        path = "results/multicar_iterative"
        os.makedirs(path, exist_ok=True)

        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        x_init = jnp.array([state_init.pipeline_state])
        r_terms_all_global = jnp.array([jnp.zeros((args.n_robots, 7))])  # shape (1, n, 6)
        state = state_init
        r_terms_all = []
        for t in range(U_init.shape[0]):
            u_t = jnp.clip(U_init[t], -1.0, 1.0)
            state = step_env_jit(state, U_init[t])
            x_init = jnp.concatenate([x_init, state.pipeline_state[None]], axis=0)
            # Compute split rewards
            _, r_terms = env.get_rewards(state.pipeline_state, u_t)  # r_terms: (n, 6)
            r_terms_all.append(r_terms)
            r_terms_all_global = jnp.concatenate([r_terms_all_global, r_terms[None]], axis=0)

        x_init = jnp.transpose(x_init, (1, 0, 2))
        r_terms_all = jnp.stack(r_terms_all, axis=1)  # shape (n, T, 6)
        # Save final trajectory plot
        r_total_from_terms = r_terms_all[:, :, -1]         # shape (n, T)
        r_total_mean = r_total_from_terms.mean(axis=-1)
        print("From get_rewards:", r_total_mean)

        r_terms_all_global = jnp.transpose(r_terms_all_global, (1, 0, 2))  # (n, T+1, 6)

        # === Initial rollout (U_init) ===
        state_init = reset_env_jit(jax.random.PRNGKey(args.seed + 1024))
        xs = jnp.array([state_init.pipeline_state])
        state = state_init
        r_terms_all_local = jnp.array([jnp.zeros((args.n_robots, 7))])  # shape (1, n, 6)
        for t in range(U_opt.shape[0]):
            u_t = U_opt[t]
            state = step_env_jit(state, U_opt[t])
            xs = jnp.concatenate([xs, state.pipeline_state[None]], axis=0)
            _, r_terms = env.get_rewards(state.pipeline_state, u_t)  # r_terms: (n, 6)
            r_terms_all_local = jnp.concatenate([r_terms_all_local, r_terms[None]], axis=0)
        xs = jnp.transpose(xs, (1, 0, 2))

        r_terms_all_local = jnp.transpose(r_terms_all_local, (1, 0, 2))  # (n, T+1, 6)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.set_aspect('equal', adjustable='datalim')

        env.render(ax, xs, goals=env.xg, actions=U_opt)

        ecd_tag = "LIDEC" if args.ECD else "LID"
        formation_tag = "_form" if args.formation_shift else ""

        # === Second figure: env.render only ===
        fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
        ax2.set_aspect('equal', adjustable='datalim')
        env.render(ax2, xs, goals=env.xg, actions=U_opt)
        plt.title(f"local reverse diffusion-{ecd_tag}{formation_tag}")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"local_reverse_diffusion{ecd_tag}.pdf"))
        print(f"Figure saved in {path}/local_diffusion_{ecd_tag}.pdf")

        # ================== FIGURE A: GLOBAL REVERSE DIFFUSION ONLY ================== #
        fig_g, ax_g = plt.subplots(1, 1, figsize=(5, 5))
        ax_g.set_aspect('equal', adjustable='datalim')
        # Use env.render with X = x_init and, optionally, global actions U_init
        env.render(ax_g, x_init, goals=env.xg, actions=U_init, style="flow")
        ax_g.set_title(f"Global reverse diffusion")
        plt.tight_layout()
        plt.savefig(os.path.join(path, f"global_diffusion.pdf"))
        print(f"Global figure saved in {path}/global_diffusion.pdf")

        output_dir_local = os.path.join(path, "reward_local")
        output_dir_global = os.path.join(path, "reward_global")
        component_names = ["r_goal", "r_safe", "r_form", "r_control", "r_obs", "r_backward", "r_total"]
        plot_and_save_reward_terms(r_terms_all_local, path=output_dir_local, prefix=f"reward_plot_{ecd_tag}_{formation_tag}", component_names=component_names)
        plot_and_save_reward_terms(r_terms_all_global, path=output_dir_global, prefix=f"globalreward_plot_{ecd_tag}_{formation_tag}", component_names=component_names)

        # Save trajectory data and optimized control
        np.savez(f"results/multicar_iterative/optimized_data_{ecd_tag}_{formation_tag}.npz", U_opt=np.array(U_opt), traj=np.array(traj), goals=np.array(env.xg), rewards=np.array(rewards_per_iter))

        plot_all_robot_actions(U_opt)

        # Create video

        if args.high_resolution:
            print("Interpolating high resolution trajectory for smoother rendering...")
            xs_interp, t_interp = interpolate_trajectory_jax(xs, dt_original=0.1, dt_interp=0.01)
            xs_np = jnp.array(xs_interp)
        else:
            print("Using original resolution trajectory for rendering...")
            xs_np = jnp.array(xs)

        n, T, _ = xs_np.shape
        cmap = plt.get_cmap('tab20', n)
        actions = jnp.transpose(U_opt, (1, 0, 2))  # (n, T, Nu)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal', adjustable='box')

        lines = [ax.plot([], [], 'o', label=f"Robot {i}")[0] for i in range(n)]
        goals = env.xg if hasattr(env, "xg") else None

        # Compute bounding box
        x_all = xs_np[:, :, 0].flatten()
        y_all = xs_np[:, :, 1].flatten()

        x_min, x_max = x_all.min(), x_all.max()
        y_min, y_max = y_all.min(), y_all.max()

        margin = 2

        ax.set_xlim(float(x_min - margin), float(x_max + margin))
        ax.set_ylim(float(y_min - margin), float(y_max + margin))

        ax.set_title("Robot tracking")

        if args.formation_shift:
            c0 = env.x0[:, :2].mean(axis=0)
            circle0 = plt.Circle((c0[0], c0[1]), env.radius, color='gray', linestyle='--', fill=False)
            ax.add_patch(circle0)
            cg = env.xg[:, :2].mean(axis=0)
            circleg = plt.Circle((cg[0], cg[1]), env.radius, color='black', linestyle='--', fill=False)
            ax.add_patch(circleg)

        ax.legend()

        def init():
            for line in lines:
                line.set_data([], [])
            return lines

        # Create line and point objects for each robot
        lines = []
        points = []

        for i in range(n):
            color = cmap(i)
            line, = ax.plot([], [], lw=2, color=color, zorder=1)
            lines.append(line)

            if goals is not None:
                    gx, gy = goals[i, 0], goals[i, 1]
                    ax.plot(gx, gy, 's', color=color, markersize=6, markeredgewidth=2)

        orientation_arrows = []
        robot_circles = []
        def update(frame):
            # Remove previous arrows
            for arr in orientation_arrows:
                arr.remove()
            orientation_arrows.clear()
            # Remove previous circles
            for c in robot_circles:
                c.remove()
            robot_circles.clear()
            for i in range(n):
                x_trail = xs_np[i, :frame + 1, 0]
                y_trail = xs_np[i, :frame + 1, 1]
                lines[i].set_data(x_trail, y_trail)

                x_curr = xs_np[i, frame, 0]
                y_curr = xs_np[i, frame, 1]
                theta_curr = xs_np[i, frame, 2]

                circle = plt.Circle((x_curr, y_curr), 0.1, color=cmap(i), fill=False, linestyle='-', linewidth=1, zorder=3)
                ax.add_patch(circle)
                robot_circles.append(circle)

                dx = 0.3 * np.cos(theta_curr)
                dy = 0.3 * np.sin(theta_curr)

                v_curr = actions[i, frame, 1]
                color_arrow = 'green' if v_curr >= -0.05 else 'red'

                arrow = ax.arrow(x_curr, y_curr, dx, dy, head_width=0.1, head_length=0.15, fc=color_arrow, ec=color_arrow)

                orientation_arrows.append(arrow)

            return lines + points + orientation_arrows

        ani = animation.FuncAnimation(
            fig, update, frames=T, init_func=init, blit=False, interval=100
        )
        labels = [f"Robot {i}" for i in range(n)]
        buffer_min = 0.2
        buffer_max = 0.5

        for x_c, y_c, w, h in env.static_obstacles:
            rect_outer = plt.Rectangle(
                (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
                w + 2 * buffer_max,
                h + 2 * buffer_max,
                linewidth=0,
                facecolor='yellow',
                alpha=0.1,
                zorder=1
            )
            ax.add_patch(rect_outer)

        for x_c, y_c, w, h in env.static_obstacles:
            rect_inner = plt.Rectangle(
                (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
                w + 2 * buffer_min,
                h + 2 * buffer_min,
                linewidth=0,
                facecolor='yellow',
                alpha=0.5,
                zorder=2
            )
            ax.add_patch(rect_inner)

        for x_c, y_c, w, h in env.static_obstacles:
            rect_real = plt.Rectangle(
                (x_c - w / 2, y_c - h / 2),
                w, h,
                linewidth=1,
                edgecolor='red',
                facecolor='red',
                zorder=3
            )
            ax.add_patch(rect_real)

        ax.legend(points, labels, loc='upper right')

        plot_obstacle_layout(env)

        video_path = os.path.join(path, f"local_diffusion_{ecd_tag}_{formation_tag}.mp4")
        ani.save(video_path, fps=10, dpi=150)
        print("Video saved in:", video_path)



if __name__ == "__main__":
    main()
