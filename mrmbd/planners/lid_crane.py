from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import time
import os
import tyro
import jax.debug
from mrmbd.envs.class_carroponte import CranePendulumEnv, Args, rollout_single_us
from mrmbd.utils import cosine_beta_schedule
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


def animate_crane_pendulum(x_traj, l, dt, tag="global", inputs=None, speedup=1, blit=True, show=False):
    """
    Animation of the overhead crane with pendulum on a horizontal rail.
    """

    # --- Sanity check ---
    x_traj = np.asarray(x_traj)
    assert x_traj.ndim == 2 and x_traj.shape[1] == 4, \
        f"x_traj expected shape (T,4) [theta,dtheta,x,dx], got {x_traj.shape}"
    T = len(x_traj)

    # Remap: [x, dx, theta, dtheta]
    states = np.empty((T, 4), dtype=float)
    states[:, 0] = x_traj[:, 2]  # x
    states[:, 1] = x_traj[:, 3]  # dx
    states[:, 2] = x_traj[:, 0]  # theta
    states[:, 3] = x_traj[:, 1]  # dtheta

    class Animation:
        def __init__(self, states, inputs, dt, ax, pend_length):
            self.states = states
            self.inputs = None if inputs is None else np.asarray(inputs).reshape(-1)
            self.dt = dt
            self.ax = ax

            # Dimensions
            self.cart_width = 0.4
            self.cart_height = 0.2
            self.wheel_radius = 0.025
            self.pend_length = pend_length

            # Artists (will be created on first update)
            self.cart = None
            self.wheel0 = None
            self.wheel1 = None
            self.rod = None
            self.force = None
            self.time_txt = self.ax.text(
                0.02, 0.95, "time = 0.00 s",
                transform=self.ax.transAxes, ha="left", va="top"
            )

        def init(self):
            # Ground line
            self.ax.plot(
                [-100, 100],
                [-self.cart_height/2 - self.wheel_radius*2,
                 -self.cart_height/2 - self.wheel_radius*2],
                linewidth=1, color='k'
            )
            return []

        def update(self, frame):
            x, dx, th, dth = self.states[frame]

            # Create artists if they don't exist yet
            if self.cart is None:
                self.cart = self.ax.add_patch(
                    Rectangle((x - self.cart_width/2, -self.cart_height/2),
                              self.cart_width, self.cart_height,
                              linewidth=1, edgecolor='k', facecolor='grey')
                )
                self.wheel0 = self.ax.add_patch(
                    Circle((x - self.cart_width/4, -self.cart_height/2 - self.wheel_radius),
                           self.wheel_radius, linewidth=1, edgecolor='k', facecolor='w')
                )
                self.wheel1 = self.ax.add_patch(
                    Circle((x + self.cart_width/4, -self.cart_height/2 - self.wheel_radius),
                           self.wheel_radius, linewidth=1, edgecolor='k', facecolor='w')
                )
                self.rod, = self.ax.plot([], [], 'o-k', lw=2)
                self.force = self.ax.annotate(
                    "", xy=(0, 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color='red')
                )

            # Update cart and wheels
            self.cart.set_x(x - self.cart_width/2)
            self.wheel0.center = (x - self.cart_width/4, -self.cart_height/2 - self.wheel_radius)
            self.wheel1.center = (x + self.cart_width/4, -self.cart_height/2 - self.wheel_radius)

            # Pendulum
            tip_x = x + self.pend_length * np.sin(th)
            tip_y = self.pend_length * np.cos(th)
            self.rod.set_data([x, tip_x], [0.0, tip_y])

            # Time label
            self.time_txt.set_text(f"time = {frame*self.dt:.2f} s")

            return [self.cart, self.wheel0, self.wheel1, self.rod, self.force, self.time_txt]

    # === Figure setup ===
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title(f"Crane Pendulum - Trajectory [{tag}]")

    anim = Animation(states, inputs, dt, ax, pend_length=l)

    interval_ms = dt * 1000.0 / max(1, speedup)
    fps_save = max(1, int(round(1.0 / dt)))

    ani = FuncAnimation(
        fig=fig,
        func=anim.update,
        init_func=anim.init,
        frames=T,
        interval=interval_ms,
        blit=blit
    )

    # Save video
    out_dir = "results/crane"
    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, f"carroponte_{tag}.mp4")
    ani.save(video_path, fps=fps_save, dpi=200)

    if show:
        plt.show()
    plt.close(fig)
    print(f"Video saved to {video_path}")


def run_diffusion_once(args: Args, env, rollout_us_fn, reset_env_jit):
    rng = jax.random.PRNGKey(args.seed)

    Nu = env.action_size

    rng, rng_reset = jax.random.split(rng)
    state_init = reset_env_jit(rng_reset)

    # Diffusion noise schedule
    betas = jnp.linspace(args.beta0, args.betaT, args.Ndiffuse)
    alphas = 1.0 - betas
    alphas_bar = jnp.cumprod(alphas)
    sigmas = jnp.sqrt(1.0 - alphas_bar)
    YN = jnp.zeros([args.Hsample, Nu])
    times = []

    def reverse_once(carry):
        i, rng, Ybar_i = carry
        t0 = time.perf_counter()

        Yi = Ybar_i * jnp.sqrt(alphas_bar[i])

        rng, rng_eps = jax.random.split(rng)
        eps_u = jax.random.normal(rng_eps, (args.Nsample, args.Hsample, Nu))

        Y0s = eps_u * sigmas[i] + Ybar_i
        Y0s = jnp.clip(Y0s, -1, 1)
        rewss, pipeline_state, r_terms = jax.vmap(rollout_us_fn)(Y0s)
        rews = rewss.mean(axis=-1)

        rew_std = rews.std()
        rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)

        logp0 = (rews - rews.mean()) / rew_std / args.temp_sample

        weights = jax.nn.softmax(logp0)
        Ybar = jnp.einsum("s,shj->hj", weights, Y0s)

        score = 1 / (1.0 - alphas_bar[i]) * (-Yi + jnp.sqrt(alphas_bar[i]) * Ybar)
        Yim1 = 1 / jnp.sqrt(alphas[i]) * (Yi + (1.0 - alphas_bar[i]) * score)
        Ybar_im1 = Yim1 / jnp.sqrt(alphas_bar[i - 1])

        jax.debug.print("Step {}: mean reward = {:.4f}, std = {:.4f}", i, rews.mean(), rew_std)
        t1 = time.perf_counter()
        times.append(t1 - t0)

        return (i - 1, rng, Ybar_im1), Yi, Y0s

    def reverse(YN, rng):
        Yi = YN
        for i in reversed(range(1, args.Ndiffuse)):
            carry = (i, rng, Yi)
            (i, rng, Yi), Yi_current, Y0s = reverse_once(carry)
        return Yi

    rng_exp, rng = jax.random.split(rng)
    U_0 = reverse(YN, rng_exp)

    return U_0


def run_diffusion_local(args: Args, U_init: jnp.ndarray, env, rollout_us_fn, reset_env_jit):
    times = []

    rng = jax.random.PRNGKey(args.seed + 123)
    H = args.Hsample
    Nu = env.action_size
    L = 50  # window size
    K = 2   # number of iterations
    rewards_per_iter = []

    U = U_init.copy()
    betas = jnp.linspace(args.beta0, args.betaT, 10)
    alphas = 1.0 - betas
    alphas_bar_local = jnp.cumprod(alphas)
    sigmas_local = jnp.sqrt(1 - alphas_bar_local)
    times_local = []

    for k in range(K):
        for t_start in range(0, H - L + 1, L // 2):
            t_k_start = time.time()
            t_end = t_start + L
            U_window = U[t_start:t_end]
            rng, rng_step = jax.random.split(rng)

            @jax.jit
            def reverse_once_local(U_w, rng_w):
                local_times = []
                for j in reversed(range(1, 10)):
                    t0 = time.perf_counter()
                    eps_u = jax.random.normal(rng_w, (args.Nsample, L, Nu))
                    sigma_local = sigmas_local[j]
                    Y0s = eps_u * sigma_local + U_w

                    Y0s = jnp.clip(Y0s, -1, 1)  # Clip to action bounds

                    # Insert Y0s into full trajectory
                    U_fulls = jnp.repeat(U[None, ...], args.Nsample, axis=0)  # (Nsample, H, Nu)
                    U_fulls = U_fulls.at[:, t_start:t_end, :].set(Y0s)
                    state_init = reset_env_jit(rng_step)
                    rewss, _, _ = jax.vmap(rollout_us_fn)(U_fulls)

                    rews = rewss.mean(axis=-1)

                    logp0 = (rews - rews.mean()) / (rews.std() + 1e-6) / args.temp_sample
                    weights = jax.nn.softmax(logp0)
                    U_opt = jnp.einsum("s,slj->lj", weights, Y0s)

                    U_new = jnp.sqrt(alphas_bar_local[j - 1]) * U_opt

                    t1 = time.perf_counter()
                    local_times.append(t1 - t0)

                times_local.extend(local_times)
                return U_new, _

            t0 = time.perf_counter()
            U_opt_local, _ = reverse_once_local(U_window, rng_step)
            t1 = time.perf_counter()
            times.append(t1 - t0)

            U = U.at[t_start:t_end, :].set(U_opt_local)

        t_k_end = time.time()
        iter_time = t_k_end - t_k_start
        iter_freq = 1.0 / iter_time if iter_time > 0 else 0.0
        print(f"Mean time per local window = {np.mean(times)*1000:.2f} ms")

        state_init_eval = reset_env_jit(jax.random.PRNGKey(args.seed))
        rewss_eval, _, _ = rollout_us_fn(U)
        reward_mean = rewss_eval.mean()
        rewards_per_iter.append(float(reward_mean))
        print(f"[Iteration {k}] reward = {reward_mean:.4f} | time = {iter_time:.3f}s | freq = {iter_freq:.2f} Hz")

    return U, rewards_per_iter

def main():
    args = tyro.cli(Args)
    env = CranePendulumEnv(dt=0.04)

    step_env_jit = jax.jit(env.step)
    reset_env_jit = jax.jit(env.reset)
    state_init = reset_env_jit(jax.random.PRNGKey(args.seed))

    rollout_us_fn = jax.jit(partial(rollout_single_us, step_env_jit, state_init))

    print("== Phase 1: Initial reverse diffusion ==")
    U_init = run_diffusion_once(args, env, rollout_us_fn, reset_env_jit)

    print("== Phase 2: Iterative local optimization ==")
    U_opt, _ = run_diffusion_local(args=args, U_init=U_init, env=env, rollout_us_fn=rollout_us_fn, reset_env_jit=reset_env_jit)

    print("== Final rollout and visualization ==")
    rewards, x_traj, r_terms = rollout_us_fn(U_init)
    rewards_opt, x_traj, r_terms_opt = rollout_us_fn(U_opt)
    state = reset_env_jit(jax.random.PRNGKey(args.seed))
    states = []
    for t in range(U_init.shape[0]):
        u_t = U_init[t]
        state = step_env_jit(state, u_t)
        states.append(state.pipeline_state)
    x_init = jnp.stack(states, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.set_aspect('equal', adjustable='datalim')

    state = reset_env_jit(jax.random.PRNGKey(args.seed))
    states = []
    for t in range(U_init.shape[0]):
        u_t = U_opt[t]
        state = step_env_jit(state, u_t)
        states.append(state.pipeline_state)

    x_opt = jnp.stack(states, axis=0)
    fig1, ax1 = plt.subplots(1, 1, figsize=(5, 5))
    ax1.set_aspect('equal', adjustable='datalim')

    env.render(x_init, U=U_init, r_terms=r_terms, rewards=rewards, tag="global")
    env.render(x_opt, U=U_opt, r_terms=r_terms_opt, rewards=rewards_opt, tag="local")

    # Animation
    animate_crane_pendulum(x_init, env.l, env.dt, tag="global")
    animate_crane_pendulum(x_opt, env.l, env.dt, tag="local")

    # Extended metrics
    # === Compute metrics ===
    theta = x_opt[:, 0]
    x = x_opt[:, 2]

    theta_ref = 0.0        # pendulum upright
    x_ref = env.qf[2]     # cart final position

    t = np.arange(0, len(x_opt)*env.dt, env.dt)
    err_theta = np.arctan2(np.sin(theta_ref - theta), np.cos(theta_ref - theta))
    err_x = x_ref - x
    err0 = abs(np.arctan2(np.sin(theta_ref - theta[0]), np.cos(theta_ref - theta[0]))) + 1e-6

    MSE_theta = np.sqrt(np.mean(err_theta**2))
    MAE_theta = np.mean(np.abs(err_theta))
    Einf_theta = np.max(np.abs(err_theta))
    MSE_x = np.mean(err_x**2)
    MAE_x = np.mean(np.abs(err_x))
    Einf_x = np.max(np.abs(err_x))

    tol = 0.05
    band_theta = tol * err0
    band_x = tol * (1.0 if x_ref == 0 else abs(x_ref))

    def settling_time(err, t, band, N=10):
        for i in range(len(err)-N):
            if np.all(np.abs(err[i:i+N]) < band):
                return t[i]
        return np.nan

    Ts_theta = settling_time(err_theta, t, band_theta)
    Ts_x = settling_time(err_x, t, band_x)

    overshoot_theta = (np.max(np.abs(err_theta)) - err0)/err0 * 100
    overshoot_x = np.max(np.abs(x - x_ref)) * 100

    # === Actual control values ===
    U_real = U_opt.squeeze() * env.max_u  # real scale [N]
    control_energy = np.sum(U_real**2)*env.dt
    control_max = np.max(np.abs(U_real))

    print(f"MSE theta = {MSE_theta:.4f}, MAE theta = {MAE_theta:.4f}, |e|_inf = {Einf_theta:.4f}")
    print(f"MSE x     = {MSE_x:.4f}, MAE x     = {MAE_x:.4f}, |e|_inf = {Einf_x:.4f}")
    print(f"Ts theta  = {Ts_theta:.2f} s, Ts x = {Ts_x:.2f} s")
    print(f"Overshoot theta = {overshoot_theta:.2f} %, Overshoot x = {overshoot_x:.2f} %")
    print(f"Control energy = {control_energy:.3f}, Max force = {control_max:.2f} N")
    print(f"Max force = {control_max} N")

    # === Energy and plots ===
    out_dir = "results/crane"
    os.makedirs(out_dir, exist_ok=True)

    dpos = np.gradient(x_opt[:, 2], env.dt)
    dtheta = np.gradient(x_opt[:, 0], env.dt)

    E_kin = 0.5*env.M*dpos**2 + 0.5*env.m*((dpos + env.l*dtheta*np.cos(theta))**2 + (env.l*dtheta*np.sin(theta))**2)
    E_pot = env.m*env.g*env.l*(np.cos(theta))
    E_tot = E_kin + E_pot

    # === Angular error plot ===
    plt.figure()
    plt.plot(t, err_theta)
    plt.xlabel("Time [s]")
    plt.ylabel("Error theta [rad]")
    plt.title("Angular error over time")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "errore_theta_diffusion.png"), dpi=200)

    # === Energy plot ===
    plt.figure()
    plt.plot(t, E_kin, label="Kinetic energy")
    plt.plot(t, E_pot, label="Potential energy")
    plt.plot(t, E_tot, '--', label="Total energy")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.title("Energy over time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "energie_diffusion.png"), dpi=200)

    # === Control force plot ===
    plt.figure()
    plt.plot(t, U_real)
    plt.xlabel("Time [s]")
    plt.ylabel("Force [N]")
    plt.title("Control force over time")
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, "forza_diffusion.png"), dpi=200)

    # === State plots ===
    plt.figure(figsize=(6, 5))
    plt.subplot(2, 1, 1)
    plt.plot(t, x)
    plt.ylabel("Cart position [m]")
    plt.grid(True)
    theta_wrapped = np.arctan2(np.sin(theta), np.cos(theta))
    plt.subplot(2, 1, 2)
    plt.plot(t, theta_wrapped)
    plt.ylabel("Pendulum angle [rad]")
    plt.xlabel("Time [s]")
    plt.grid(True)
    plt.suptitle("System state evolution (Diffusion)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stati_diffusion.png"), dpi=200)
    plt.close('all')

if __name__ == "__main__":
    main()
