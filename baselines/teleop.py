"""Delayed-feedback controller: the surface-pilot proxy, and the task oracle.

Two jobs, one class.

1. **The baseline.** With a latency budget it models a pilot on a ship: stale
   observations, commands issued at a low rate and held, plus a reaction lag.
   This is what the autonomy result is measured against.

2. **The oracle.** With zero latency and a fast command rate it is simply a
   well-tuned PD controller with full state access. If *that* cannot complete
   the task, the task is impossible and no amount of PPO will fix it. We run
   it first for exactly that reason.

Honesty about what this is
--------------------------
It is a delayed PD controller, not a human. A real pilot has feedforward
skill a PD lacks -- they learn to anticipate a periodic swell. A real pilot
also fatigues, loses the picture in zero visibility, and cannot attend to
four vehicles at once, none of which this models. It is therefore neither a
strict lower nor a strict upper bound on human performance. It isolates one
variable -- loop latency -- which is the variable under test.

Gains are tuned by the same search for every latency setting (see
`tune_gains`), so the baseline is never handicapped by bad tuning. A delayed
loop needs *lower* gain to stay stable, and letting it keep the fast-loop
gains would manufacture instability and overstate our own result.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import math
import torch

from sim.env import RepairEnv, LIFT, STABILISE, MATE


@dataclass
class TeleopSpec:
    """Latency budget of one control mode."""
    name: str
    obs_delay_ms: float = 600.0     # round trip to the surface
    cmd_hz: float = 2.0             # command update rate, held between
    reaction_ms: float = 250.0      # operator reaction lag
    kp_xy: float = 0.55
    kd_xy: float = 1.10
    kp_z: float = 0.70
    kd_z: float = 1.30
    kp_yaw: float = 0.80
    kd_yaw: float = 0.60
    ff_heave: float = 0.22          # feedforward hold for cable weight

    @property
    def total_delay_s(self) -> float:
        """Effective loop delay, including the command-hold half-period."""
        return (self.obs_delay_ms + self.reaction_ms) / 1000.0 \
            + 0.5 / self.cmd_hz

    @property
    def corner_hz(self) -> float:
        """Achievable closed-loop bandwidth, f = 1/(2*pi*L)."""
        return 1.0 / (2.0 * math.pi * max(self.total_delay_s, 1e-6))


MODES = {
    "teleop_600": TeleopSpec("teleop_600", 600.0, 2.0, 250.0,
                             kp_xy=0.22, kd_xy=0.55, kp_z=0.30, kd_z=0.70,
                             kp_yaw=0.35, kd_yaw=0.30),
    "teleop_150": TeleopSpec("teleop_150", 150.0, 2.0, 250.0,
                             kp_xy=0.38, kd_xy=0.80, kp_z=0.50, kd_z=1.00,
                             kp_yaw=0.55, kd_yaw=0.45),
    # zero latency, full rate: the oracle / solvability check
    "pd_ideal": TeleopSpec("pd_ideal", 0.0, 50.0, 0.0),
}


class TeleopController:
    """Batched delayed PD controller over a `RepairEnv`.

    Delay is implemented as a ring buffer of past *observations of the world*
    (not of the action), because that is where a pilot's latency actually
    lives: they are looking at an old picture.
    """

    def __init__(self, env: RepairEnv, spec: TeleopSpec) -> None:
        self.env = env
        self.spec = spec
        self.dt = env.sim.control_dt
        delay_steps = int(round(
            (spec.obs_delay_ms + spec.reaction_ms) / 1000.0 / self.dt))
        self.delay_steps = max(0, delay_steps)
        self.cmd_period = max(1, int(round(1.0 / (spec.cmd_hz * self.dt))))
        self.buf: deque = deque(maxlen=self.delay_steps + 1)
        self.held = torch.zeros(env.batch, env.n_rov, 4, device=env.device,
                                dtype=env.dtype)
        self.k = 0

    def _snapshot(self) -> dict:
        """What the pilot would see: positions, velocities, targets."""
        e = self.env
        m_err, m_relv, m_dyaw = e.mate_error()
        _, _, heading = e._mate_frame()
        return dict(pos=e.swarm.pos.clone(), vel=e.swarm.vel.clone(),
                    yaw=e.swarm.yaw.clone(), yaw_rate=e.swarm.yaw_rate.clone(),
                    tgt=e.targets(), phase=e.phase.clone(),
                    m_err=m_err.clone(), m_relv=m_relv.clone(),
                    heading=heading.clone())

    def reset(self) -> None:
        self.buf.clear()
        self.held.zero_()
        self.k = 0

    def act(self) -> torch.Tensor:
        """Return the command for this control step."""
        self.buf.append(self._snapshot())
        # Refresh the command only on a command tick; hold it otherwise.
        # This zero-order hold is a real part of the latency, and omitting it
        # would understate the baseline's delay by half a command period.
        if self.k % self.cmd_period == 0 or self.buf.maxlen == 1:
            s = self.buf[0] if len(self.buf) == self.buf.maxlen else self.buf[0]
            self.held = self._pd(s)
        self.k += 1
        return self.held

    @staticmethod
    def _to_body(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """Rotate a world-frame vector into each vehicle's body frame.

        Thruster surge/sway act along the hull, not along world north/east.
        A world-frame error used directly as a surge/sway demand drives the
        vehicle sideways by exactly the yaw angle -- invisible while yaw is
        near zero, and divergent for the connector vehicle, which actively
        yaws to align with the cable.
        """
        c, sn = torch.cos(yaw), torch.sin(yaw)
        bx = c * vec[..., 0] + sn * vec[..., 1]
        by = -sn * vec[..., 0] + c * vec[..., 1]
        return torch.stack([bx, by, vec[..., 2]], dim=-1)

    def _pd(self, s: dict) -> torch.Tensor:
        sp = self.spec
        yaw = s["yaw"]
        err = self._to_body(s["tgt"] - s["pos"], yaw)
        vel = self._to_body(s["vel"], yaw)
        a = torch.zeros_like(self.held)
        a[..., 0] = sp.kp_xy * err[..., 0] - sp.kd_xy * vel[..., 0]
        a[..., 1] = sp.kp_xy * err[..., 1] - sp.kd_xy * vel[..., 1]
        a[..., 2] = sp.kp_z * err[..., 2] - sp.kd_z * vel[..., 2] + sp.ff_heave

        # yaw: lifters hold heading, the connector vehicle aligns to the cable
        dyaw = torch.zeros_like(yaw)
        dyaw[:, 3] = torch.remainder(s["heading"] - yaw[:, 3] + math.pi,
                                     2.0 * math.pi) - math.pi
        a[..., 3] = sp.kp_yaw * dyaw - sp.kd_yaw * s["yaw_rate"]

        # During MATE the connector vehicle closes on the node directly, with
        # the gain scheduled down near contact so it does not overshoot the
        # 8 cm tolerance.
        in_mate = (s["phase"] == MATE)
        if bool(in_mate.any()):
            d = self._to_body(s["m_err"].unsqueeze(1), yaw[:, 3:4]).squeeze(1)
            rv = self._to_body(s["m_relv"].unsqueeze(1), yaw[:, 3:4]).squeeze(1)
            dist = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            gain = 1.2 * torch.tanh(dist / 0.25)
            cmd = -(d / dist) * gain - 0.9 * rv
            sel = in_mate.unsqueeze(-1)
            a[:, 3, :3] = torch.where(sel, cmd, a[:, 3, :3])
        return a.clamp(-1.0, 1.0)
