"""
SnakeEnv: a planar snake robot (N rigid links, N-1 actuated revolute joints)
moving on the ground with anisotropic viscous friction.  Task: reach a goal.

Physics
-------
Each link experiences viscous drag that is larger normal to the link than
along it (c_n > c_t).  That anisotropy is what lets lateral undulation
propel a snake.  In the high-friction / low-inertia regime the net ground
force and torque on the body are zero at every instant, so given the joint
velocities the body twist (vx, vy, omega) is the solution of a 3x3 linear
system (the "reduced" overdamped model used in geometric-mechanics work on
snake locomotion, e.g. Hatton & Choset).  It is exact for rigid rods under
linear anisotropic drag, cheap, deterministic and numerically robust, which
makes it well suited to RL.  Swap `_body_twist` for MuJoCo/PyBullet if you
want full rigid-body dynamics with inertia.

Action modes
------------
"cpg"    (default) action = [amplitude, frequency, spatial phase lag, turn offset]
         of a serpenoid oscillator that the joint servos track.  Much easier to
         learn, and the policy is free to modulate the gait to steer or reverse.
"direct" action = target angle for every joint (normalised to [-1, 1]).

Reward (per control step)
-------------------------
  + w_progress * (d_prev - d_now) / body_length     progress toward goal
  - w_energy   * mean((joint_vel / max_vel)^2)      actuation effort
  - w_smooth   * mean((a_t - a_{t-1})^2)            action smoothness
  - w_time                                          encourages speed
  + goal_bonus                                      on reaching the goal (terminates)
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        n_links: int = 8,
        link_length: float = 0.15,        # m
        c_t: float = 1.0,                 # tangential drag coeff (per unit length)
        c_n: float = 4.0,                 # normal drag coeff (per unit length)
        dt: float = 0.1,                  # control period (s)
        substeps: int = 5,
        max_joint_angle: float = 0.9,     # rad
        max_joint_vel: float = 4.0,       # rad/s  (servo rate limit)
        servo_gain: float = 10.0,         # 1/s    (P-gain of servo)
        action_mode: str = "cpg",
        goal_dist_range: tuple = (1.0, 3.0),   # in body lengths
        goal_radius: float = 0.2,         # m
        max_steps: int = 300,
        w_progress: float = 20.0,
        w_energy: float = 0.02,
        w_smooth: float = 0.02,
        w_time: float = 0.01,
        goal_bonus: float = 10.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert action_mode in ("cpg", "direct")
        self.n = n_links
        self.L = link_length
        self.BL = n_links * link_length
        self.c_t, self.c_n = c_t, c_n
        self.c_rot = c_n * link_length**3 / 12.0   # rod spinning about its centre
        self.dt, self.substeps = dt, substeps
        self.a_max, self.w_max, self.kp = max_joint_angle, max_joint_vel, servo_gain
        self.mode = action_mode
        self.goal_dist_range = goal_dist_range
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.w_prog, self.w_en, self.w_sm = w_progress, w_energy, w_smooth
        self.w_time, self.goal_bonus = w_time, goal_bonus
        self.render_mode = render_mode

        nj = n_links - 1
        self.nj = nj
        # CPG parameter ranges
        self.cpg_amp = (0.0, max_joint_angle)
        self.cpg_freq = (0.0, 1.5)        # Hz
        self.cpg_lag = (-1.6, 1.6)        # rad between neighbouring joints
        self.cpg_off = (-0.5, 0.5)        # rad constant bias -> steering

        self.act_dim = 4 if action_mode == "cpg" else nj
        self.action_space = spaces.Box(-1.0, 1.0, (self.act_dim,), np.float32)
        obs_dim = 2 * nj + 3 + 3 + self.act_dim + (2 if action_mode == "cpg" else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)

        self._fig = None
        self.reset_state_vars()

    # ------------------------------------------------------------------ physics
    def reset_state_vars(self):
        self.pose = np.zeros(3)                  # x, y, theta of link 0 (head)
        self.alpha = np.zeros(self.nj)
        self.alpha_dot = np.zeros(self.nj)
        self.phase = 0.0
        self.goal = np.zeros(2)
        self.prev_act = np.zeros(self.act_dim)
        self.body_vel = np.zeros(3)
        self.t = 0
        self.trail = []

    def _kinematics(self, pose, alpha):
        """Link centres P (n,2), link angles Theta (n,), joint positions J (n-1,2).
        Link 0 is the head and points along +Theta; later links trail behind it."""
        Theta = pose[2] + np.concatenate([[0.0], np.cumsum(alpha)])
        d = np.stack([np.cos(Theta), np.sin(Theta)], 1)
        h = 0.5 * self.L
        P = np.empty((self.n, 2))
        J = np.empty((self.n - 1, 2))
        P[0] = pose[:2]
        for i in range(self.n - 1):
            J[i] = P[i] - h * d[i]          # body trails behind the head (link 0)
            P[i + 1] = J[i] - h * d[i + 1]
        return Theta, P, J

    def _body_twist(self, pose, alpha, alpha_dot):
        """World-frame twist (vx, vy, omega) of link 0 such that net friction wrench = 0."""
        Theta, P, J = self._kinematics(pose, alpha)
        n = self.n
        c, s = np.cos(Theta), np.sin(Theta)
        t = np.stack([c, s], 1)
        nrm = np.stack([-s, c], 1)
        D = self.L * (self.c_t * t[:, :, None] * t[:, None, :]
                      + self.c_n * nrm[:, :, None] * nrm[:, None, :])        # (n,2,2)
        r = P - P[0]
        G = np.zeros((n, 2, 3))
        G[:, 0, 0] = 1.0
        G[:, 1, 1] = 1.0
        G[:, 0, 2] = -r[:, 1]
        G[:, 1, 2] = r[:, 0]
        # Velocity of link i caused purely by joint motion (link 0 held fixed):
        # joint j rotates every link i > j about the joint position J[j].
        mask = np.tril(np.ones((n, n - 1)), -1)          # mask[i, j] = 1 if i > j
        rel = P[:, None, :] - J[None, :, :]               # (n, n-1, 2)
        perp = np.stack([-rel[..., 1], rel[..., 0]], -1)
        v_sh = np.einsum("ij,ijk->ik", mask * alpha_dot[None, :], perp)      # (n,2)
        w_sh = mask @ alpha_dot                                              # (n,)
        A = np.einsum("iak,iab,ibl->kl", G, D, G)
        A[2, 2] += n * self.c_rot
        b = np.einsum("iak,iab,ib->k", G, D, v_sh)
        b[2] += self.c_rot * w_sh.sum()
        return np.linalg.solve(A, -b)

    def _servo_targets(self, a):
        if self.mode == "direct":
            return a * self.a_max
        lerp = lambda x, lim: lim[0] + 0.5 * (x + 1.0) * (lim[1] - lim[0])
        amp = lerp(a[0], self.cpg_amp)
        freq = lerp(a[1], self.cpg_freq)
        lag = a[2] * self.cpg_lag[1]
        off = a[3] * self.cpg_off[1]
        self.phase = (self.phase + 2 * np.pi * freq * self.dt) % (2 * np.pi)
        j = np.arange(self.nj)
        tgt = amp * np.sin(self.phase - j * lag) + off
        return np.clip(tgt, -self.a_max, self.a_max)

    def _com(self, pose=None, alpha=None):
        _, P, _ = self._kinematics(self.pose if pose is None else pose,
                                   self.alpha if alpha is None else alpha)
        return P.mean(0)

    def _heading(self):
        Theta = self.pose[2] + np.concatenate([[0.0], np.cumsum(self.alpha)])
        return np.arctan2(np.sin(Theta).sum(), np.cos(Theta).sum())

    # ---------------------------------------------------------------- gym API
    def _obs(self):
        com = self._com()
        hd = self._heading()
        R = np.array([[np.cos(hd), np.sin(hd)], [-np.sin(hd), np.cos(hd)]])   # world->body
        g_body = R @ (self.goal - com)
        dist = np.linalg.norm(g_body)
        err = np.arctan2(g_body[1], g_body[0])
        v_com_b = R @ self.body_vel[:2]
        o = [self.alpha / self.a_max, self.alpha_dot / self.w_max,
             [dist / self.BL, np.cos(err), np.sin(err)],
             [v_com_b[0] / self.L, v_com_b[1] / self.L, self.body_vel[2]],
             self.prev_act]
        if self.mode == "cpg":
            o.append([np.sin(self.phase), np.cos(self.phase)])
        return np.concatenate(o).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_state_vars()
        rng = self.np_random
        self.pose = np.array([0.0, 0.0, rng.uniform(-np.pi, np.pi)])
        self.alpha = rng.uniform(-0.1, 0.1, self.nj)
        self.phase = rng.uniform(0, 2 * np.pi)
        com = self._com()
        ang = rng.uniform(-np.pi, np.pi)
        dist = rng.uniform(*self.goal_dist_range) * self.BL
        self.goal = com + dist * np.array([np.cos(ang), np.sin(ang)])
        self.prev_dist = np.linalg.norm(self.goal - com)
        self.trail = [com.copy()]
        return self._obs(), {"distance": self.prev_dist}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self._servo_targets(a)
        h = self.dt / self.substeps
        energy = 0.0
        for _ in range(self.substeps):
            ad = np.clip(self.kp * (target - self.alpha), -self.w_max, self.w_max)
            new_alpha = np.clip(self.alpha + ad * h, -self.a_max, self.a_max)
            ad = (new_alpha - self.alpha) / h
            u = self._body_twist(self.pose, self.alpha, ad)
            self.pose = self.pose + h * u
            self.alpha, self.alpha_dot, self.body_vel = new_alpha, ad, u
            energy += np.mean((ad / self.w_max) ** 2) / self.substeps
        self.pose[2] = (self.pose[2] + np.pi) % (2 * np.pi) - np.pi
        self.t += 1

        com = self._com()
        self.trail.append(com.copy())
        dist = np.linalg.norm(self.goal - com)
        progress = (self.prev_dist - dist) / self.BL
        self.prev_dist = dist
        success = dist < self.goal_radius

        reward = (self.w_prog * progress
                  - self.w_en * energy
                  - self.w_sm * np.mean((a - self.prev_act) ** 2)
                  - self.w_time
                  + (self.goal_bonus if success else 0.0))
        self.prev_act = a
        terminated = bool(success)
        truncated = self.t >= self.max_steps
        info = {"distance": dist, "success": bool(success), "energy": energy}
        return self._obs(), float(reward), terminated, truncated, info

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode != "rgb_array":
            return None
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        if self._fig is None:
            self._fig = Figure(figsize=(5, 5), dpi=100)
            self._canvas = FigureCanvasAgg(self._fig)
            self._ax = self._fig.add_subplot(111)
        ax = self._ax
        ax.clear()
        _, P, J = self._kinematics(self.pose, self.alpha)
        tr = np.array(self.trail)
        pts = np.vstack([tr, P, self.goal[None]])
        c, half = pts.mean(0), max(1.0, 0.6 * np.ptp(pts, axis=0).max() + 0.4)
        ax.set_xlim(c[0] - half, c[0] + half)
        ax.set_ylim(c[1] - half, c[1] + half)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.plot(tr[:, 0], tr[:, 1], color="tab:blue", lw=1, alpha=0.5)
        ax.add_patch(__import__("matplotlib").patches.Circle(
            self.goal, self.goal_radius, color="tab:green", alpha=0.6))
        Theta = self.pose[2] + np.concatenate([[0.0], np.cumsum(self.alpha)])
        h = 0.5 * self.L
        for i in range(self.n):
            d = h * np.array([np.cos(Theta[i]), np.sin(Theta[i])])
            ax.plot(*zip(P[i] - d, P[i] + d), color="tab:red" if i == 0 else "k", lw=5,
                    solid_capstyle="round")
        ax.set_title(f"t={self.t * self.dt:.1f}s  dist={self.prev_dist:.2f}m")
        self._canvas.draw()
        return np.asarray(self._canvas.buffer_rgba())[..., :3].copy()

    def close(self):
        self._fig = None


if "Snake-v0" not in gym.envs.registry:
    gym.register(id="Snake-v0", entry_point="snake_env:SnakeEnv")
