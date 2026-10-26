"""Batched 4-DOF ROV dynamics with thruster allocation and a cable gripper.

Control interface
-----------------
The policy emits a 4-vector in [-1, 1]: normalised demand for surge, sway,
heave and yaw. That demand is *allocated* across six physical thrusters (four
vectored horizontal, two vertical -- the BlueROV-heavy layout), each with its
own force limit and first-order lag. If the allocation saturates a thruster,
the achieved wrench is clipped and will not match what was asked for.

This matters. An idealised model where the policy directly sets body forces
learns a controller that cannot exist. By forcing the demand through a real
allocation matrix with real saturation, the policy has to learn that, for
example, a hard yaw command eats the sway authority it might need a moment
later to catch a gust. That coupling is the thing a hand-tuned PID gets wrong.

Reduced-order choices (all stated in the submission's limits section):
  * roll and pitch are assumed trim-stabilised and are not modelled
  * added mass is a diagonal constant, not a frequency-dependent tensor
  * the tether to the surface (if any) is not modelled as a dynamic load
"""

from typing import Optional, Tuple

import math
import torch

from .config import ROVSpec, RHO_WATER, GRAVITY
from .water import CurrentField, quadratic_drag


def allocation_matrix(device, dtype) -> torch.Tensor:
    """[6, 4] map from thruster forces to body wrench, and its pseudo-inverse.

    Horizontal thrusters sit at +-45 degrees at the four corners of a
    0.9 x 0.7 m footprint; verticals are on the centreline. Columns are
    (surge, sway, heave, yaw).
    """
    a = math.sqrt(0.5)
    lx, ly = 0.45, 0.35
    rows = [
        # surge,  sway,  heave,  yaw
        [+a, +a, 0.0, +(lx * a + ly * a)],   # front-left,  vectored
        [+a, -a, 0.0, -(lx * a + ly * a)],   # front-right
        [+a, -a, 0.0, +(lx * a + ly * a)],   # rear-left
        [+a, +a, 0.0, -(lx * a + ly * a)],   # rear-right
        [0.0, 0.0, 1.0, 0.0],                # vertical port
        [0.0, 0.0, 1.0, 0.0],                # vertical starboard
    ]
    return torch.tensor(rows, device=device, dtype=dtype)


class ROVSwarm:
    """A batch of B environments, each with K independent ROVs.

    State is stored as [B, K, ...] so the whole swarm of every environment
    advances in one set of tensor ops.
    """

    def __init__(
        self,
        spec: ROVSpec,
        batch: int,
        n_rov: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.spec = spec
        self.batch = batch
        self.n_rov = n_rov
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (batch, n_rov)
        self.pos = torch.zeros(*shape, 3, device=self.device, dtype=dtype)
        self.vel = torch.zeros_like(self.pos)
        self.yaw = torch.zeros(*shape, device=self.device, dtype=dtype)
        self.yaw_rate = torch.zeros_like(self.yaw)

        # Commanded vs achieved thruster state, for the first-order lag.
        self.thrust = torch.zeros(*shape, 6, device=self.device, dtype=dtype)
        self.demand = torch.zeros(*shape, 4, device=self.device, dtype=dtype)

        # Which cable node, if any, each ROV is gripping. -1 = free.
        self.grip_node = torch.full(shape, -1, device=self.device,
                                    dtype=torch.long)
        self.grip_slipped = torch.zeros(*shape, device=self.device,
                                        dtype=torch.bool)

        A = allocation_matrix(self.device, dtype)
        self.A = A                                        # [6, 4]
        self.A_pinv = torch.linalg.pinv(A)                # [4, 6]

        # Per-thruster force limit. The horizontal four share the xy budget,
        # the vertical pair shares the heave budget.
        self.f_max = torch.tensor(
            [spec.max_force_xy / 2.0] * 4 + [spec.max_force_z / 2.0] * 2,
            device=self.device, dtype=dtype)

        self.m_xy = spec.eff_mass_xy
        self.m_z = spec.eff_mass_z
        self.cum_energy = torch.zeros(*shape, device=self.device, dtype=dtype)

    # ------------------------------------------------------------------ setup

    def reset(self, pos: torch.Tensor, yaw: Optional[torch.Tensor] = None,
              mask: Optional[torch.Tensor] = None) -> None:
        """Place the swarm. `mask` [B] restricts the reset to some envs."""
        if mask is None:
            self.pos[:] = pos
            self.vel.zero_()
            if yaw is None:
                self.yaw.zero_()
            else:
                self.yaw.copy_(yaw)
            self.yaw_rate.zero_()
            self.thrust.zero_()
            self.demand.zero_()
            self.grip_node.fill_(-1)
            self.grip_slipped.zero_()
            self.cum_energy.zero_()
            return
        m = mask.view(-1, 1, 1)
        m2 = mask.view(-1, 1)
        self.pos = torch.where(m, pos, self.pos)
        self.vel = torch.where(m, torch.zeros_like(self.vel), self.vel)
        new_yaw = torch.zeros_like(self.yaw) if yaw is None else yaw
        self.yaw = torch.where(m2, new_yaw, self.yaw)
        self.yaw_rate = torch.where(m2, torch.zeros_like(self.yaw_rate),
                                    self.yaw_rate)
        self.thrust = torch.where(m, torch.zeros_like(self.thrust), self.thrust)
        self.demand = torch.where(m, torch.zeros_like(self.demand), self.demand)
        self.grip_node = torch.where(m2, torch.full_like(self.grip_node, -1),
                                     self.grip_node)
        self.grip_slipped = self.grip_slipped & ~m2
        self.cum_energy = torch.where(m2, torch.zeros_like(self.cum_energy),
                                      self.cum_energy)

    # ------------------------------------------------------------ allocation

    def allocate(self, demand: torch.Tensor) -> torch.Tensor:
        """[B, K, 4] normalised wrench demand -> [B, K, 6] thruster forces, N.

        The demand is scaled to physical units, pushed through the
        pseudo-inverse of the allocation matrix, then clipped per thruster.
        Clipping is where the honest coupling comes from: we do *not*
        re-normalise to preserve direction, because a real vehicle does not
        get to choose that either.
        """
        sp = self.spec
        scale = torch.tensor(
            [sp.max_force_xy, sp.max_force_xy, sp.max_force_z,
             sp.max_torque_yaw], device=self.device, dtype=self.dtype)
        wrench = demand.clamp(-1.0, 1.0) * scale
        f = torch.einsum("ij,bkj->bki", self.A_pinv.transpose(0, 1), wrench)
        return f.clamp(-self.f_max, self.f_max)

    def achieved_wrench(self, f_thruster: torch.Tensor) -> torch.Tensor:
        """[B, K, 6] thruster forces -> [B, K, 4] body wrench actually produced."""
        return torch.einsum("ij,bki->bkj", self.A, f_thruster)

    # ------------------------------------------------------------------- step

    def step(self, dt: float, demand: torch.Tensor, current: CurrentField,
             t: torch.Tensor,
             grip_force: Optional[torch.Tensor] = None,
             seabed_z: float = 0.0) -> dict:
        """Advance every ROV by `dt` under a normalised wrench `demand`.

        `grip_force` [B, K, 3] is the reaction from the cable gripper, i.e.
        minus the force the ROV applies to the cable. Passing it in rather
        than computing it here keeps the cable the single owner of the
        gripper constraint.
        """
        sp = self.spec

        # --- demand slew limit, then first-order thruster lag
        max_step = sp.thrust_rate_limit * dt
        d_clamped = demand.clamp(-1.0, 1.0)
        self.demand = self.demand + (d_clamped - self.demand).clamp(
            -max_step, max_step)
        f_cmd = self.allocate(self.demand)
        alpha = dt / (sp.thrust_tau + dt)
        self.thrust = self.thrust + alpha * (f_cmd - self.thrust)
        wrench = self.achieved_wrench(self.thrust)        # [B, K, 4]

        # --- thruster force in the world frame (body yaw rotates surge/sway)
        c, s = torch.cos(self.yaw), torch.sin(self.yaw)
        fx_b, fy_b, fz, mz = wrench.unbind(-1)
        f_world = torch.stack([c * fx_b - s * fy_b,
                               s * fx_b + c * fy_b,
                               fz], dim=-1)

        # --- hydrodynamics
        u = current.velocity(self.pos.reshape(self.batch, -1, 3), t)
        u = u.reshape_as(self.pos)
        v_rel = self.vel - u
        cd = torch.tensor([sp.cd_xy, sp.cd_xy, sp.cd_z],
                          device=self.device, dtype=self.dtype)
        area = torch.tensor([sp.area_xy, sp.area_xy, sp.area_z],
                            device=self.device, dtype=self.dtype)
        speed = v_rel.norm(dim=-1, keepdim=True)
        f_drag = -0.5 * RHO_WATER * cd * area * speed * v_rel

        f_buoy = torch.zeros_like(self.pos)
        f_buoy[..., 2] = sp.net_buoyancy

        f_total = f_world + f_drag + f_buoy
        if grip_force is not None:
            f_total = f_total + grip_force

        m = torch.tensor([self.m_xy, self.m_xy, self.m_z],
                         device=self.device, dtype=self.dtype)
        self.vel = self.vel + dt * f_total / m
        self.pos = self.pos + dt * self.vel

        # --- seabed: ROVs do not drive through the mud
        below = self.pos[..., 2] < seabed_z + 0.5
        self.pos[..., 2] = self.pos[..., 2].clamp_min(seabed_z + 0.5)
        self.vel[..., 2] = torch.where(below & (self.vel[..., 2] < 0),
                                       torch.zeros_like(self.vel[..., 2]),
                                       self.vel[..., 2])

        # --- yaw, with quadratic hydrodynamic damping
        m_damp = -sp.damp_yaw * self.yaw_rate.abs() * self.yaw_rate
        self.yaw_rate = self.yaw_rate + dt * (mz + m_damp) / sp.inertia_yaw
        self.yaw = torch.remainder(self.yaw + dt * self.yaw_rate + math.pi,
                                   2.0 * math.pi) - math.pi

        # --- energy, for the efficiency term in the reward
        p = self.thrust.abs().sum(-1) * 0.001        # kW-ish proxy
        self.cum_energy = self.cum_energy + p * dt

        return dict(wrench=wrench, thrust=self.thrust,
                    saturated=(self.thrust.abs() >= 0.995 * self.f_max),
                    demand_tracking=(wrench[..., :3].norm(dim=-1)))

    # ---------------------------------------------------------------- gripper

    def gripper_forces(self, cable_x: torch.Tensor, cable_v: torch.Tensor,
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Spring-damper tie between each gripping ROV and its cable node.

        Returns (force_on_cable [B, N, 3], force_on_rov [B, K, 3]).

        A gripper is not a weld: above `grip_max_force` it slips, the hold is
        lost, and the mission layer has to notice and re-plan. That is the
        failure the slow loop exists to handle.
        """
        sp = self.spec
        B, N, _ = cable_x.shape
        f_cable = torch.zeros_like(cable_x)
        f_rov = torch.zeros_like(self.pos)

        held = self.grip_node >= 0
        if not bool(held.any()):
            return f_cable, f_rov

        idx = self.grip_node.clamp_min(0)                      # [B, K]
        node_x = torch.gather(cable_x, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
        node_v = torch.gather(cable_v, 1, idx.unsqueeze(-1).expand(-1, -1, 3))

        # The ROV holds the node at a standoff below itself, as a real
        # manipulator would: the vehicle is above the cable it is lifting.
        target = self.pos - torch.tensor([0.0, 0.0, 1.0],
                                         device=self.device,
                                         dtype=self.dtype) * 0.8
        err = target - node_x
        derr = self.vel - node_v
        f = sp.grip_stiffness * err + sp.grip_damping * derr

        mag = f.norm(dim=-1, keepdim=True)
        slipped = (mag.squeeze(-1) > sp.grip_max_force) & held
        f = f * (sp.grip_max_force / mag.clamp_min(1e-9)).clamp(max=1.0)
        f = f * held.unsqueeze(-1).to(self.dtype)

        self.grip_slipped = slipped
        self.grip_node = torch.where(slipped,
                                     torch.full_like(self.grip_node, -1),
                                     self.grip_node)

        f_cable = f_cable.scatter_add(
            1, idx.unsqueeze(-1).expand(-1, -1, 3), f)
        f_rov = -f
        return f_cable, f_rov
