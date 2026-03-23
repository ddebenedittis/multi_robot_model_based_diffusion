# RRPR manipulator environment (4-DOF: Revolute-Revolute-Prismatic-Revolute)

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from functools import partial
from flax import struct
from mrmbd.envs.manipulator import B_func_jax, C_func_jax, G_func_jax
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib
import tyro
from mpl_toolkits.mplot3d import Axes3D

matplotlib.use('Agg')


def plot_four_panel(Y, t, titles, ylabels, suptitle, outfile):
    fig, axs = plt.subplots(4, 1, figsize=(10.5, 8), sharex=True)
    for i in range(4):
        axs[i].plot(t, Y[:, i], linewidth=1.5)
        axs[i].set_title(titles[i], fontsize=11)
        axs[i].set_ylabel(ylabels[i])
        axs[i].grid(True, alpha=0.3)
    axs[-1].set_xlabel("Step")
    fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    plt.savefig(f"{outfile}.png", dpi=300)
    plt.savefig(f"{outfile}.pdf")
    plt.close()


os.makedirs("results/manipulator", exist_ok=True)


def dh_matrix(theta, d, a, alpha):
    ct, st = jnp.cos(theta), jnp.sin(theta)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    return jnp.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0]
    ])


@dataclass
class Args:
    seed: int = 42
    Nsample: int = 4096
    Hsample: int = 100
    Ndiffuse: int = 100
    beta0: float = 1e-3
    betaT: float = 1e-4
    temp_sample: float = 0.1
    save_video: bool = False
    Nsample_local = 64


@partial(jax.jit, static_argnums=(1))
def get_joint_positions(q, param):
    theta1, theta2, d3, theta4 = q
    a1, a2, a3, a4 = param[:, 0]
    alpha1, alpha2, alpha3, alpha4 = param[:, 1]
    d1, d2, _, d4 = param[:, 2]

    T01 = dh_matrix(theta1, d1, a1, alpha1)
    T12 = dh_matrix(theta2, d2, a2, alpha2)
    T23 = dh_matrix(0, d3, a3, alpha3)
    T34 = dh_matrix(theta4, d4, a4, alpha4)

    points = [np.zeros(3)]
    T = np.eye(4)
    for T_next in [T01, T12, T23, T34]:
        T = T @ T_next
        point = np.array(T[:3, 3]).reshape(3,)
        points.append(point)

    return np.stack(points, axis=0)  # shape (5, 3)


@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def forward_kinematics_rrpr_jax(q, L1, L2, L3, L4, D2):
    """Forward kinematics for RRPR manipulator using DH convention."""
    pi = jnp.pi
    theta1 = q[0]
    theta2 = q[1]
    theta4 = q[3]
    d3 = q[2]
    a = [L1, L2, L3, L4]
    alpha = [0, pi, 0, 0]
    d = [0, 0, d3, D2]

    T01 = dh_matrix(theta1, d[0], a[0], alpha[0])
    T12 = dh_matrix(theta2, d[1], a[1], alpha[1])
    T23 = dh_matrix(0.0, d[2], a[2], alpha[2])
    T34 = dh_matrix(theta4, d[3], a[3], alpha[3])

    T04 = T01 @ T12 @ T23 @ T34
    return T04, T01, T12, T23, T34


def angle_diff(q, qf):
    return jnp.arctan2(jnp.sin(q - qf), jnp.cos(q - qf))


def rollout_single_us(step_env, state, us):
    def step_fn(state, u_t):
        state = step_env(state, u_t)
        return state, (state.reward, state.pipeline_state, state.r_terms)

    _, (rews, states, r_terms) = jax.lax.scan(step_fn, state, us)
    states = jnp.vstack([state.pipeline_state[None], states])
    rews = jnp.hstack([0.0, rews])
    return rews, states, r_terms


@struct.dataclass
class State:
    pipeline_state: jnp.ndarray  # state: [q1,q2,q3,q4, dq1,dq2,dq3,dq4]
    reward: float
    r_terms: jnp.ndarray


def rk4(dynamics, x, u, dt):
    k1 = dynamics(x, u)
    k2 = dynamics(x + dt / 2 * k1, u)
    k3 = dynamics(x + dt / 2 * k2, u)
    k4 = dynamics(x + dt * k3, u)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@jax.jit
def B_func_jitted(q, m, L1, L2, L3, L4, D2):
    return B_func_jax(q, m, L1, L2, L3, L4, D2)


@jax.jit
def C_func_jitted(q, dq, m, L1, L2, L3, L4, D2):
    return C_func_jax(q, dq, m, L1, L2, L3, L4, D2)


@jax.jit
def G_func_jitted(q, m, g, L1, L2, L3, L4):
    return G_func_jax(q, m, g, L1, L2, L3, L4)


class RRPRSingleEnv:
    def __init__(self, dt=0.001):
        self.dt = dt
        self.H = 100
        self.ACTION_SCALE = jnp.array([175, 50, 75, 2])
        self.q0 = jnp.hstack([jnp.array([0.1, 0.1, 0.1, 0.1]), jnp.zeros(4)])
        self.qf = jnp.hstack([jnp.array([-0.8, 0.8, 0.03, 0.8]), jnp.zeros(4)])

        self.L1_num = 0.40
        self.L2_num = 0.30
        self.L3_num = 0.0
        self.D2_num = 0.10
        self.L4_num = self.D2_num

        self.m_num = jnp.array([6.0, 4.0, 1.0, 0.8])
        self.g0_num = jnp.array([0., 0., -9.81])

        self.a = jnp.array([self.L1_num, self.L2_num, self.L3_num, self.L4_num])
        self.alpha = jnp.array([0, jnp.pi, 0, 0])
        self.d = jnp.array([0, 0, 0, self.D2_num])
        self.param = jnp.array([self.a, self.alpha, self.d]).T  # shape (4, 3)
        self.q_min = jnp.array([-jnp.inf, -jnp.inf, 0.0, -jnp.inf])
        self.q_max = jnp.array([jnp.inf, jnp.inf, 0.3, jnp.inf])

    def rrpr_dynamics(self, x, u):
        q = x[:4]
        dq = x[4:]

        B = B_func_jitted(q, self.m_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        C = C_func_jitted(q, dq, self.m_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        G = G_func_jitted(q, self.m_num, self.g0_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num)
        G = G.squeeze()

        ddq = jnp.linalg.solve(B, u - C @ dq - G)
        return jnp.concatenate([dq, ddq])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng):
        return State(pipeline_state=self.q0, reward=0.0, r_terms=jnp.zeros(5))

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: State, action: jax.Array) -> State:
        action = jnp.clip(action, -1.0, 1.0)
        action_scaled = action * self.ACTION_SCALE
        q_new = rk4(self.rrpr_dynamics, state.pipeline_state, action_scaled, self.dt)
        reward, r_terms = self.get_rewards(q_new, action_scaled)
        return State(pipeline_state=q_new, reward=reward, r_terms=r_terms)

    @partial(jax.jit, static_argnums=(0,))
    def get_rewards(self, q, u):
        pos = q[:4]
        vel = q[4:]

        # Joint-space error (normalized to initial error)
        angle_errors_1 = angle_diff(self.qf[0], pos[0])
        angle_errors_2 = angle_diff(self.qf[1], pos[1])
        angle_errors_3 = angle_diff(self.qf[3], pos[3])
        linear_error = self.qf[2] - pos[2]

        angle_errors_1_0 = angle_diff(self.qf[0], self.q0[0])
        angle_errors_2_0 = angle_diff(self.qf[1], self.q0[1])
        angle_errors_3_0 = angle_diff(self.qf[3], self.q0[3])
        linear_error_0 = self.qf[2] - self.q0[2]

        joint_error = jnp.sqrt(angle_errors_1**2 + angle_errors_2**2 + angle_errors_3**2)
        joint_error_0 = jnp.sqrt(angle_errors_1_0**2 + angle_errors_2_0**2 + angle_errors_3_0**2)

        r_q_goal = (
            0.5 * (1.0 - joint_error / (joint_error_0 + 1e-6)) +
            0.5 * (1.0 - jnp.abs(linear_error) / (jnp.abs(linear_error_0) + 1e-6))
        )

        # End-effector error (normalized to initial error)
        T_curr, *_ = forward_kinematics_rrpr_jax(pos, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        ee_pos = T_curr[:3, 3]

        T_goal, *_ = forward_kinematics_rrpr_jax(self.qf[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        ee_goal = T_goal[:3, 3]

        err = jnp.linalg.norm(ee_goal - ee_pos)

        T_start, *_ = forward_kinematics_rrpr_jax(self.q0[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
        ee_start = T_start[:3, 3]
        err0 = jnp.maximum(jnp.linalg.norm(ee_goal - ee_start), 1e-8)

        r_goal = 1.0 - (err / err0)

        # Gravity compensation reward
        tau_required = G_func_jitted(pos, self.m_num, self.g0_num, self.L1_num, self.L2_num, self.L3_num, self.L4_num)
        delta_tau = u - tau_required.squeeze()

        # Exponential gains (max when vel=0 and tau=tau_required)
        k_vel = 20.0
        k_tau = 10
        gain_vel = jnp.exp(-k_vel * jnp.sum(vel**2))
        gain_tau = jnp.exp(-k_tau * jnp.sum(delta_tau**2))

        r_total = 10 * r_q_goal + 10 * r_goal + gain_tau + gain_vel

        r_terms = jnp.array([r_goal, r_q_goal, gain_vel, gain_tau, r_total])
        return r_total, r_terms

    @property
    def action_size(self):
        return 4

    @property
    def observation_size(self):
        return 8

    @property
    def num_robots(self):
        return 1

    def render(self, X: jnp.ndarray, tau_seq: jnp.ndarray, rewards: jnp.ndarray = None,
               r_terms: jnp.ndarray = None, tag: str = ""):
        """Save static plots and 3D animation of the RRPR manipulator."""
        os.makedirs("results/manipulator_diffusion", exist_ok=True)

        X = np.array(X)
        tau_seq = np.array(tau_seq)
        q_seq = X[:, :4]
        dq_seq = X[:, 4:]
        T = len(X)
        step = np.arange(T)

        # Joint positions
        plot_four_panel(
            Y=q_seq, t=step,
            titles=["q1", "q2", "q3", "q4"],
            ylabels=["[rad]", "[rad]", "[m]", "[rad]"],
            suptitle="Joint positions",
            outfile=f"results/manipulator_diffusion/q_all_{tag}")

        # Joint velocities
        plot_four_panel(
            Y=dq_seq, t=step,
            titles=["dq1", "dq2", "dq3", "dq4"],
            ylabels=["[rad/s]", "[rad/s]", "[m/s]", "[rad/s]"],
            suptitle="Joint velocities",
            outfile=f"results/manipulator_diffusion/dq_all_{tag}")

        # Control torques
        plot_four_panel(
            Y=tau_seq, t=step,
            titles=[r"$\tau_1$", r"$\tau_2$", r"$\tau_3$", r"$\tau_4$"],
            ylabels=["[Nm]", "[Nm]", "[N]", "[Nm]"],
            suptitle="Control torques/force",
            outfile=f"results/manipulator_diffusion/tau_all_{tag}")

        # Reward terms
        if r_terms is not None:
            labels = ["r_goal", "r_q_goal", "r_vel", "r_tau"]
            plot_four_panel(
                Y=r_terms[:, :4], t=step,
                titles=labels, ylabels=["[-]"] * 4,
                suptitle="Reward components per step",
                outfile=f"results/manipulator_diffusion/reward_terms_{tag}")

            plt.figure(figsize=(10.5, 3))
            plt.plot(step, r_terms[:, 4], linewidth=1.6)
            plt.xlabel("Step"); plt.ylabel("[-]"); plt.title("Total reward")
            plt.grid(True, alpha=0.3); plt.tight_layout()
            plt.savefig(f"results/manipulator_diffusion/reward_total_{tag}.png", dpi=300)
            plt.savefig(f"results/manipulator_diffusion/reward_total_{tag}.pdf")
            plt.close()

        # EE error over time
        ee_errors = []
        for t in range(T):
            q = q_seq[t]
            T_curr, *_ = forward_kinematics_rrpr_jax(q, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            ee_pos = np.array(T_curr[:3, 3])
            T_goal, *_ = forward_kinematics_rrpr_jax(self.qf[:4], self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            ee_goal = np.array(T_goal[:3, 3])
            ee_errors.append(np.linalg.norm(ee_goal - ee_pos))

        plt.figure(figsize=(10, 4))
        plt.plot(ee_errors, label="EE error to goal", color='tab:red')
        plt.xlabel("Step"); plt.ylabel("Error [m]")
        plt.title(f"End-effector error to goal ({tag})")
        plt.grid(True); plt.tight_layout()
        plt.savefig(f"results/manipulator_diffusion/ee_error_{tag}.png", dpi=300)
        plt.close()

        # 3D animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim([-0.8, 0.8]); ax.set_ylim([-0.8, 0.8]); ax.set_zlim([-0.5, 0.3])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.view_init(elev=45, azim=45); ax.grid(True)

        robot_line, = ax.plot([], [], [], 'ko-', linewidth=2)
        trail_line, = ax.plot([], [], [], 'b--', linewidth=1.5)
        goal_point, = ax.plot([], [], [], 'rx', markersize=8, label="Goal")
        title3d = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        ax.legend()

        trail_x, trail_y, trail_z = [], [], []

        def update(frame):
            nonlocal trail_x, trail_y, trail_z

            q = q_seq[frame]
            qf = np.array(self.qf[:4])

            T04, T01, T12, T23, T34 = forward_kinematics_rrpr_jax(q, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            T02 = T01 @ T12; T03 = T02 @ T23; T04 = T03 @ T34

            p0 = jnp.array([0, 0, 0])
            p1 = T01[:3, 3]; p2 = T02[:3, 3]; p3 = T03[:3, 3]; p4 = T04[:3, 3]
            points = np.stack([p0, p1, p2, p3, p4], axis=0)
            ee_pos = np.array(p4)

            T_goal, *_ = forward_kinematics_rrpr_jax(qf, self.L1_num, self.L2_num, self.L3_num, self.L4_num, self.D2_num)
            ee_goal = np.array(T_goal[:3, 3])

            robot_line.set_data(points[:, 0], points[:, 1])
            robot_line.set_3d_properties(points[:, 2])
            goal_point.set_data([ee_goal[0]], [ee_goal[1]])
            goal_point.set_3d_properties([ee_goal[2]])

            trail_x.append(ee_pos[0]); trail_y.append(ee_pos[1]); trail_z.append(ee_pos[2])
            trail_line.set_data(trail_x, trail_y)
            trail_line.set_3d_properties(trail_z)

            title3d.set_text(f"Frame {frame}")
            return robot_line, trail_line, goal_point, title3d

        ani = FuncAnimation(fig, update, frames=T, interval=50)
        ani.save(f"results/manipulator_diffusion/motion_3D_{tag}.mp4", writer='ffmpeg', fps=20)
        plt.close(fig)
