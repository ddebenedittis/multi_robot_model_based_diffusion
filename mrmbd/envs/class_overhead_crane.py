# Crane-pendulum swing-up environment
# Convention: theta = 0 at top (unstable equilibrium), theta = pi at bottom

import os
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import struct


@struct.dataclass
class Args:
    seed: int = 42
    Nsample: int = 4096
    Hsample: int = 200
    Ndiffuse: int = 100
    beta0: float = 1e-3
    betaT: float = 1e-4
    temp_sample: float = 0.1


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # state: [theta, dtheta, x, dx]
    reward: float
    r_terms: jnp.ndarray


def rollout_single_us(step_env, state, us):
    """Execute a rollout from an initial state with a sequence of controls."""

    def step_fn(state, u_t):
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state, state.r_terms)

    _, (rews, states, r_terms) = jax.lax.scan(step_fn, state, us)

    # Prepend initial state
    states = jnp.vstack([state.pipeline_state[None], states])
    rews = jnp.hstack([0.0, rews])

    return rews, states, r_terms


class CranePendulumEnv:
    def __init__(self, dt=0.04):
        self.dt = dt
        self.H = 200

        # Physical parameters
        self.M = 10.0  # cart mass [kg]
        self.m = 1.0  # pendulum mass [kg]
        self.l = 1.0  # pendulum length [m]
        self.g = 9.81  # gravity [m/s^2]
        self.b = 0.0  # friction

        # Maximum force
        self.max_u = 50.0

        # Initial state (pendulum near bottom)
        self.q0 = jnp.array([0.9 * jnp.pi, 0.0, 0.0, 0.0])
        # Goal state (pendulum at top)
        self.qf = jnp.array([0.0, 0.0, -0.8, 0.0])

    def dynamics(self, q, u):
        theta, dtheta, x, dx = q

        denom = self.M + self.m * jnp.sin(theta) ** 2

        theta_ddot = (
            self.M * self.g * jnp.sin(theta)
            + self.b * dx * jnp.cos(theta)
            + self.g * self.m * jnp.sin(theta)
            - 0.5 * self.l * self.m * dtheta**2 * jnp.sin(2 * theta)
            - u * jnp.cos(theta)
        ) / (self.l * denom)

        x_ddot = (
            -self.b * dx
            - 0.5 * self.g * self.m * jnp.sin(2 * theta)
            + self.l * self.m * dtheta**2 * jnp.sin(theta)
            + u
        ) / denom

        return jnp.array(
            [jnp.squeeze(dtheta), jnp.squeeze(theta_ddot), jnp.squeeze(dx), jnp.squeeze(x_ddot)]
        )

    def rk4(self, f, x, u, dt):
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        return State(pipeline_state=self.q0, reward=0.0, r_terms=jnp.zeros(3))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        u = action * self.max_u
        q_next = self.rk4(self.dynamics, state.pipeline_state, u, self.dt)
        r, r_terms = self.get_reward(q_next, u)
        return State(pipeline_state=q_next, reward=r, r_terms=r_terms)

    def get_reward(self, q, u):
        theta, dtheta, x, dx = q

        # Angular error relative to top (theta = 0)
        angle_diff = jnp.arctan2(jnp.sin(self.qf[0] - theta), jnp.cos(self.qf[0] - theta))
        linear_error = self.qf[2] - x

        error_0 = jnp.pi
        linear_error_0 = self.qf[2] - self.q0[2]
        error_angle = jnp.sqrt(angle_diff**2)
        error_linear = jnp.sqrt(linear_error**2)
        r_goal = 0.8 * (1 - error_angle / error_0 + 1e-6) + 0.2 * (
            1 - jnp.abs(error_linear) / (jnp.abs(linear_error_0) + 1e-6)
        )

        E_kin = 0.5 * self.M * dx**2 + 0.5 * self.m * (
            (dx + self.l * dtheta * jnp.cos(theta)) ** 2 + (self.l * dtheta * jnp.sin(theta)) ** 2
        )
        E_pot = self.m * self.g * self.l * jnp.cos(theta)

        r_total = 10 * r_goal
        r_total = jnp.squeeze(r_total)
        return r_total, jnp.array([jnp.squeeze(r_goal), jnp.squeeze(E_kin), jnp.squeeze(E_pot)])

    @property
    def action_size(self):
        return 1

    @property
    def observation_size(self):
        return 4

    def render(self, x_traj, U=None, rewards=None, r_terms=None, tag="", out_dir="results/crane"):
        """Visualize states, control, energies, and reward for the crane-pendulum system."""
        x_traj = jax.device_get(x_traj)
        T = x_traj.shape[0]
        t = jnp.linspace(0, T * self.dt, T)

        theta = x_traj[:, 0]
        dtheta = x_traj[:, 1]
        x = x_traj[:, 2]
        dx = x_traj[:, 3]

        figs_dir = os.path.join(out_dir, "figs")
        os.makedirs(figs_dir, exist_ok=True)

        # 1. States
        fig, axs = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
        fig.suptitle(f"Crane pendulum - States [{tag}]")
        axs[0].plot(t, theta)
        axs[0].set_ylabel("theta (rad)")
        axs[1].plot(t, dtheta)
        axs[1].set_ylabel("dtheta/dt (rad/s)")
        axs[2].plot(t, x)
        axs[2].set_ylabel("x (m)")
        axs[3].plot(t, dx)
        axs[3].set_ylabel("dx/dt (m/s)")
        axs[3].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(f"{figs_dir}/states_{tag}.png")
        plt.close()

        # 2. Control
        if U is not None:
            U = jax.device_get(U).squeeze()
            t_u = jnp.linspace(0, T * self.dt, U.shape[0])
            plt.figure(figsize=(6, 3))
            plt.plot(t_u, 50 * U)
            plt.title(f"Control force [{tag}]")
            plt.xlabel("Time (s)")
            plt.ylabel("F (N)")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{figs_dir}/control_{tag}.png")
            plt.close()

        # 3. Energies and reward terms
        if r_terms is not None:
            r_terms = jax.device_get(r_terms)
            r_goal = r_terms[:, 0]
            E_kin = r_terms[:, 1]
            E_pot = r_terms[:, 2]

            fig, axs = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
            fig.suptitle(f"Reward terms [{tag}]", fontsize=14)
            axs[0].plot(t, r_goal, color="tab:green")
            axs[0].set_ylabel("r_goal")
            axs[0].set_title("Goal tracking")
            axs[0].grid(True)
            axs[1].plot(t, E_kin, color="tab:orange")
            axs[1].set_ylabel("E_kin")
            axs[1].set_title("Kinetic energy")
            axs[1].grid(True)
            axs[2].plot(t, E_pot, color="tab:red")
            axs[2].set_ylabel("E_pot")
            axs[2].set_title("Potential energy")
            axs[2].set_xlabel("Time (s)")
            axs[2].grid(True)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.savefig(f"{figs_dir}/r_terms_{tag}.png")
            plt.close()

            # Energy plot
            E_tot = E_kin + E_pot
            plt.figure(figsize=(8, 6))
            plt.plot(t, E_kin, label="Kinetic energy", color="tab:blue", linewidth=2)
            plt.plot(t, E_pot, label="Potential energy", color="tab:orange", linewidth=2)
            plt.plot(t, E_tot, label="Total energy", color="tab:green", linestyle="--", linewidth=2)
            plt.title("Energy evolution over time", fontsize=14)
            plt.xlabel("Time [s]")
            plt.ylabel("Energy [J]")
            plt.grid(True)
            plt.legend(fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{figs_dir}/energies_{tag}.png", dpi=300)
            plt.close()

        # 4. Total reward
        if rewards is not None:
            rewards = jax.device_get(rewards)
            t_r = jnp.linspace(0, rewards.shape[0] * self.dt, rewards.shape[0])
            plt.figure(figsize=(6, 3))
            plt.plot(t_r, rewards)
            plt.title(f"Total reward [{tag}]")
            plt.xlabel("Time (s)")
            plt.ylabel("Reward")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{figs_dir}/reward_{tag}.png")
            plt.close()

        # 5. Angular error relative to unstable equilibrium (theta=0)
        delta_theta = jnp.mod(theta + jnp.pi, 2 * jnp.pi) - jnp.pi
        plt.figure(figsize=(6, 3))
        plt.plot(t, delta_theta, label="Angular error (rad)")
        plt.axhline(0, color="k", linestyle="--", lw=1)
        plt.title(f"Error relative to unstable equilibrium (theta=0) [{tag}]")
        plt.xlabel("Time (s)")
        plt.ylabel("e_theta (rad)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{figs_dir}/error_theta_{tag}.png")
        plt.close()
