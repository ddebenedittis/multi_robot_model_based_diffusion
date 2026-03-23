import jax
from jax import numpy as jnp
from flax import struct
from functools import partial
import matplotlib.pyplot as plt
from dataclasses import dataclass


# Command-line arguments for multi-car experiments
@dataclass
class Args:
    seed: int = 0
    n_robots: int = 4
    Nsample: int = 4096         # number of samples
    Hsample: int = 100          # horizon
    Ndiffuse: int = 100         # number of diffusion steps
    temp_sample: float = 0.1    # temperature for sampling
    beta0: float = 1e-4         # initial noise
    betaT: float = 1e-2         # final noise
    initial_sigma: float = 0.02 # initial gaussian noise for ECD
    alpha: float = 0.01         # optimization step size
    mu: float = 10              # penalty term
    noise_decay: float = 0.03   # decay factor for noise
    not_render: bool = False
    high_resolution: bool = False
    ECD: bool = False
    formation_shift: bool = False
    T: int = 30
    save_video: bool = False
    cosine: bool = False
    obstacles_enabled: bool = False
    penalize_backward: bool = False
    filter: bool = False


def car_dynamics(x, u):
    return jnp.array(
        [
            u[1] * jnp.cos(x[2]) * 3.0,  # x_dot
            u[1] * jnp.sin(x[2]) * 3.0,  # y_dot
            u[0] * jnp.pi,                # theta_dot
        ]
    )


# Numerical integration using Runge-Kutta 4
def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# Check for inter-robot collisions
def check_inter_robot_collisions(X_t, Ra):
    pos = X_t[:, :2]
    dists = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    collision_matrix = dists < 2 * Ra
    collision_matrix = collision_matrix.at[jnp.diag_indices(pos.shape[0])].set(False)
    return bool(jnp.any(collision_matrix))


# Generate initial and goal positions for n robots arranged in antipodal pairs on a circle
def antipodal_positions(n, radius):
    angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    x0_xy = jnp.stack([
        radius * jnp.cos(angles),
        radius * jnp.sin(angles)
    ], axis=1)
    xg_xy = -x0_xy

    # Initial orientation: pointing toward the goal
    delta = xg_xy - x0_xy
    theta0 = jnp.arctan2(delta[:, 1], delta[:, 0])
    theta_g = jnp.arctan2(delta[:, 1], delta[:, 0])

    x0 = jnp.hstack([x0_xy, theta0[:, None]])
    xg = jnp.hstack([xg_xy, theta_g[:, None]])
    return x0, xg


# Shift the entire robot formation while maintaining relative positions on the circle
def circular_shift_goals(n, radius, shift=(0.0, 3.0)):
    angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
    C_start = jnp.array([0.0, 0.0])
    C_goal = C_start + jnp.array(shift)

    x0_xy = C_start + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    xg_xy = C_goal + radius * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1)
    directions = xg_xy - x0_xy
    theta0 = jnp.arctan2(directions[:, 1], directions[:, 0])

    x0 = jnp.hstack([x0_xy, theta0[:, None]])
    xg = jnp.hstack([xg_xy, theta0[:, None]])

    return x0, xg


def check_collision_static(pos, obstacles):
    def single_obs_check(obs):
        xc, yc, w, h = obs
        x_min = xc - w / 2
        x_max = xc + w / 2
        y_min = yc - h / 2
        y_max = yc + h / 2
        dx = jnp.maximum(jnp.maximum(x_min - pos[0], 0), pos[0] - x_max)
        dy = jnp.maximum(jnp.maximum(y_min - pos[1], 0), pos[1] - y_max)
        dist = jnp.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        return dist < 0.4

    return jnp.any(jax.vmap(single_obs_check)(obstacles))


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # current state of the robot (x, y, theta)
    obs: jnp.ndarray             # observations
    reward: jnp.ndarray          # reward for each robot
    done: jnp.ndarray            # goal reached flag


class MultiCar2d:
    def __init__(self, n, radius=2.0, robot_radius=0.1, formation_shift=False,
                 obstacles_enabled=False, ECD=False, penalize_backward=False):
        self.n = n
        self.dt = 0.1
        self.H = 100
        self.Ra = robot_radius
        self.radius = radius
        self.wt = 2  # weight for safety reward
        self.formation_shift = formation_shift
        self.obstacles_enabled = obstacles_enabled
        self.ECD = ECD
        self.penalize_backward = penalize_backward

        if obstacles_enabled:
            self.static_obstacles = jnp.array([
                [0.0,  0.8, 1.5, 0.07],   # horizontal wall top
                [0.0, -0.8, 1.8, 0.07],   # horizontal wall bottom
                [-0.8, 0.0, 0.07, 1.8],   # vertical wall left
                [ 0.8, 0.0, 0.07, 1.8],   # vertical wall right
            ]) / 1.3
        else:
            self.static_obstacles = jnp.zeros((0, 4))

        if self.formation_shift:
            self.x0, self.xg = circular_shift_goals(n, radius=self.radius, shift=(0, 3.0))
        else:
            self.x0, self.xg = antipodal_positions(n, radius=self.radius)

    def reset(self, rng):
        return State(
            pipeline_state=self.x0,
            obs=self.x0,
            reward=jnp.zeros((self.n,)),
            done=jnp.zeros((self.n,)),
        )

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        q = state.pipeline_state

        # Compute the next state using Runge-Kutta 4
        q_new = jax.vmap(rk4, in_axes=(None, 0, 0, None))(
            car_dynamics, q, action, self.dt
        )

        reward, _ = self.get_rewards(q_new, action)

        return state.replace(pipeline_state=q_new, obs=q_new, reward=reward, done=jnp.zeros((self.n,)))

    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(self, q_all, u_all):
        def single_reward(k, q, u_k):
            p = q[:2]
            pT = self.xg[k][:2]
            p0 = self.x0[k][:2]
            r_goal = 1.0 - jnp.linalg.norm(p - pT) / jnp.linalg.norm(p0 - pT)

            theta = q[2]
            theta_target = self.xg[k][2]

            # Collision avoidance
            dists = jnp.linalg.norm(p - q_all[:, :2], axis=1)
            r_safe = -1.0 * jnp.any((dists <= 2 * self.Ra + 1e-2) & (jnp.arange(self.n) != k))

            # Formation cost
            def rews_formation(q_all, x0_all):
                pos = q_all[:, :2]
                pos0 = x0_all[:, :2]
                diff = pos[:, None, :] - pos[None, :, :]
                diff0 = pos0[:, None, :] - pos0[None, :, :]
                dists = jnp.linalg.norm(diff, axis=-1)
                dists0 = jnp.linalg.norm(diff0, axis=-1)
                mask = jnp.triu(jnp.ones((self.n, self.n), dtype=bool), k=1)
                return jnp.mean((dists - dists0) ** 2 * mask)

            # Static obstacle penalty
            def single_obs_penalty(p, obstacle):
                x_c, y_c, w, h = obstacle
                obs_x_min = x_c - w / 2
                obs_x_max = x_c + w / 2
                obs_y_min = y_c - h / 2
                obs_y_max = y_c + h / 2
                dx = jnp.maximum(jnp.maximum(obs_x_min - p[0], 0.0), p[0] - obs_x_max)
                dy = jnp.maximum(jnp.maximum(obs_y_min - p[1], 0.0), p[1] - obs_y_max)
                dist = jnp.sqrt(dx**2 + dy**2 + 1e-6)
                x_clipped = jnp.minimum(dist / 0.5, 1.0)
                barrier = jnp.log(x_clipped) / jnp.log(2 * self.Ra / 0.5)
                penalty = -100 * barrier
                return penalty

            if self.obstacles_enabled:
                obstacles = jnp.array(self.static_obstacles)
                r_obs_vals = jax.vmap(lambda obs: single_obs_penalty(p, obs))(obstacles)
                r_obstacles = jnp.mean(r_obs_vals)
            else:
                r_obstacles = 0.0

            if self.penalize_backward:
                v = u_k[1]
                r_backward = jnp.where(v < -0.05, -jnp.abs(v), 0.0)
                dist_to_goal = jnp.linalg.norm(p - self.xg[k][:2])
                orient_error = 1.0 - jnp.cos(theta - theta_target)
                w_orient = jnp.exp(-10.0 * dist_to_goal)
                r_orient_final = -w_orient * orient_error
            else:
                r_backward = 0.0
                r_orient_final = 0.0

            r_form = -rews_formation(q_all, self.x0) if self.formation_shift else 0.0
            r_control = -jnp.sum(u_k ** 2)
            r_total_check = r_goal + self.wt * r_safe + r_form + 0.01 * r_control + r_obstacles + r_backward + self.wt * r_orient_final
            r_terms = jnp.array([r_goal, r_safe, r_form, r_control, r_obstacles, r_backward, r_total_check])
            return r_total_check, r_terms

        r_total_all, r_terms_all = jax.vmap(single_reward, in_axes=(0, 0, 0))(jnp.arange(self.n), q_all, u_all)
        return r_total_all, r_terms_all

    @property
    def action_size(self):
        return 2

    @property
    def observation_size(self):
        return 3

    @property
    def num_robots(self):
        return self.n

    def render(self, ax, X: jnp.ndarray, goals: jnp.ndarray = None, actions: jnp.ndarray = None, style: str = "flow"):
        """Visualize optimized robot trajectories with directional arrows."""
        n = X.shape[0]

        if style == "flow":
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

        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        ]
        colors = palette[:n]

        for i in range(n):
            traj = jnp.array(X[i])
            color = colors[i % len(colors)]

            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=1.3, alpha=0.8, zorder=2)

            # Directional arrows
            step = max(2, len(traj) // 30)
            for t in range(0, len(traj) - 1, step):
                x, y, theta = traj[t, 0], traj[t, 1], traj[t, 2]
                dx = 0.1 * jnp.cos(theta)
                dy = 0.1 * jnp.sin(theta)
                ax.arrow(x, y, dx, dy,
                         head_width=0.05, head_length=0.06,
                         fc=color, ec=color, lw=0.8,
                         alpha=0.7, overhang=0.5,
                         length_includes_head=True, zorder=3)

            # Start point (white circle with colored border)
            ax.plot(traj[0, 0], traj[0, 1], marker='o', color=color, markersize=4.5,
                    markerfacecolor='white', markeredgewidth=0.9, zorder=4)
            ax.grid(True, linestyle="-", color="k", linewidth=0.6, alpha=0.7)

        # Goal markers
        if goals is not None:
            for i in range(n):
                gx, gy = goals[i, 0], goals[i, 1]
                ax.plot(gx, gy, marker='s', color=palette[i], markersize=5.5,
                        markeredgewidth=0.8, zorder=5)

        # Static obstacles
        for x_c, y_c, w, h in self.static_obstacles:
            rect = plt.Rectangle(
                (x_c - w / 2, y_c - h / 2), w, h,
                linewidth=1.0, edgecolor='black', facecolor='#d3d3d3', zorder=1)
            ax.add_patch(rect)

        # Penalty zones around obstacles
        buffer_min = 0.2
        buffer_max = 0.5
        for x_c, y_c, w, h in self.static_obstacles:
            rect_outer = plt.Rectangle(
                (x_c - (w / 2 + buffer_max), y_c - (h / 2 + buffer_max)),
                w + 2 * buffer_max, h + 2 * buffer_max,
                linewidth=0.8, edgecolor='none', facecolor='#a6bddb', alpha=0.25, zorder=1)
            ax.add_patch(rect_outer)
            rect_inner = plt.Rectangle(
                (x_c - (w / 2 + buffer_min), y_c - (h / 2 + buffer_min)),
                w + 2 * buffer_min, h + 2 * buffer_min,
                linewidth=0.8, edgecolor='none', facecolor='#3690c0', alpha=0.35, zorder=2)
            ax.add_patch(rect_inner)

        # Formation circles
        if self.formation_shift:
            c0 = self.x0[:, :2].mean(axis=0)
            cg = self.xg[:, :2].mean(axis=0)
            ax.add_patch(plt.Circle(c0, self.radius, color='gray', linestyle='--', fill=False, lw=0.6, alpha=0.5))
            ax.add_patch(plt.Circle(cg, self.radius, color='black', linestyle='--', fill=False, lw=0.6, alpha=0.5))

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Optimized trajectories", pad=8)

        ax.legend(
            [plt.Line2D([], [], color=colors[i], lw=1.3) for i in range(n)],
            [f"R{i}" for i in range(n)],
            loc='upper right', frameon=False,
            handlelength=1.5, handletextpad=0.6,
            borderpad=0.2, labelspacing=0.3,
            title="Robot", title_fontsize=9)
        ax.margins(0.1)
