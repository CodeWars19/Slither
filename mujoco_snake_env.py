"""
MujocoSnakeEnv: same task, observations, actions and reward as SnakeEnv, but the
physics is a real MuJoCo rigid-body model (with inertia, joint servos, damping).

What is MuJoCo and what is ours
-------------------------------
* MuJoCo: rigid-body dynamics, joint position actuators, integration.
* Ours:   the ground friction.  MuJoCo contact friction is isotropic in the
          tangent plane, and an undulating snake needs high-normal / low-tangent
          friction.  So we model the snake top-down (no gravity, no contacts) and
          each physics step apply the same anisotropic viscous drag as SnakeEnv
          to every link through `data.xfrc_applied`.

Because the drag model is shared, you can prototype in the fast kinematic
SnakeEnv and then retrain / sim-to-sim check here.  Policies do NOT transfer
directly (MuJoCo has inertia and different actuator dynamics): retrain, or
fine-tune with domain randomisation of c_t, c_n, mass and servo gains.
"""
from __future__ import annotations

import numpy as np
import mujoco
import gymnasium as gym

from snake_env import SnakeEnv


def build_xml(n, L, mass, radius, kp, kv, a_max, timestep, joint_damping, armature):
    h = L / 2
    geom = (f'<geom type="capsule" fromto="{-h} 0 0 {h} 0 0" size="{radius}" '
            f'mass="{mass}" contype="0" conaffinity="0" rgba="0.2 0.2 0.2 1"/>')
    # head: planar root joints (x, y, yaw) at the link centre
    root = ('<joint name="x" type="slide" axis="1 0 0"/>'
            '<joint name="y" type="slide" axis="0 1 0"/>'
            '<joint name="yaw" type="hinge" axis="0 0 1"/>')
    # build nested chain from the tail inwards
    for j in range(n - 2, -1, -1):
        child = (f'<body name="link{j + 1}" pos="{-L} 0 0">'
                 f'<joint name="q{j}" type="hinge" axis="0 0 1" pos="{h} 0 0" '
                 f'range="{-a_max} {a_max}" damping="{joint_damping}" armature="{armature}"/>'
                 f'{geom}{body_below if j < n - 2 else ""}</body>')
        body_below = child
    actuators = "".join(
        f'<position joint="q{j}" kp="{kp}" kv="{kv}" ctrlrange="{-a_max} {a_max}"/>'
        for j in range(n - 1))
    return f"""<mujoco model="snake">
  <compiler angle="radian"/>
  <option timestep="{timestep}" integrator="implicitfast" gravity="0 0 0"/>
  <worldbody>
    <body name="link0" pos="0 0 0">{root}{geom}{body_below}</body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>"""


class MujocoSnakeEnv(SnakeEnv):
    def __init__(self, link_mass: float = 0.05, radius: float = 0.02,
                 servo_kp: float = 2.0, servo_kv: float = 0.05,
                 joint_damping: float = 0.005, armature: float = 1e-4,
                 timestep: float = 0.002, c_t: float = 5.0, c_n: float = 20.0, **kw):
        super().__init__(c_t=c_t, c_n=c_n, **kw)
        xml = build_xml(self.n, self.L, link_mass, radius, servo_kp, servo_kv,
                        self.a_max, timestep, joint_damping, armature)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.n_substeps = int(round(self.dt / timestep))
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"link{i}")
                         for i in range(self.n)]
        self._vel6 = np.zeros(6)

    # ---- drag: identical formula to SnakeEnv._body_twist's friction model -------
    def _apply_drag(self):
        m, d = self.model, self.data
        d.xfrc_applied[:] = 0.0
        for bid in self.body_ids:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, self._vel6, 0)
            w, v = self._vel6[2], self._vel6[3:5]                    # yaw rate, planar velocity
            tvec = d.xmat[bid].reshape(3, 3)[:2, 0]                  # link axis in world
            nvec = np.array([-tvec[1], tvec[0]])
            f = -self.L * (self.c_t * tvec * (tvec @ v) + self.c_n * nvec * (nvec @ v))
            d.xfrc_applied[bid, 0:2] = f
            d.xfrc_applied[bid, 5] = -self.c_rot * w

    def _sync(self):
        d = self.data
        self.pose = d.qpos[:3].copy()
        self.alpha = d.qpos[3:].copy()
        self.alpha_dot = d.qvel[3:].copy()
        self.body_vel = d.qvel[:3].copy()

    def _advance(self, target):
        d = self.data
        d.ctrl[:] = target
        energy = 0.0
        for _ in range(self.n_substeps):
            self._apply_drag()
            mujoco.mj_step(self.model, d)
            energy += np.mean((d.qvel[3:] / self.w_max) ** 2) / self.n_substeps
        self._sync()
        return energy

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)       # samples pose / alpha / goal
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = self.pose
        self.data.qpos[3:] = self.alpha
        mujoco.mj_forward(self.model, self.data)
        self._sync()
        return self._obs(), {"distance": self.prev_dist}


if "Snake-MuJoCo-v0" not in gym.envs.registry:
    gym.register(id="Snake-MuJoCo-v0", entry_point="mujoco_snake_env:MujocoSnakeEnv")
