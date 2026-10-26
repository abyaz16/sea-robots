"""The repair task as a batched, gym-style environment.

Scope of the evaluated task
---------------------------
Episodes begin with the lifter ROVs already attached to the cable on the
seabed, and run LIFT -> STABILISE -> MATE. Transit to the fault and gripper
closure are initialised rather than learned.

That is deliberate, not a shortcut. The claim under test is about *holding a
loaded cable steady against disturbance* -- which is where loop latency bites
-- and bundling in a transit phase that any controller completes would only
dilute the measurement. It also means teleop and autonomy are compared on
exactly the part of the task where they differ.

Phases
------
  LIFT       raise the cable from the seabed to working height
  STABILISE  hold the span steady enough to work on
  MATE       bring the connector to the mate node and hold it in tolerance

Roles: ROVs 0-2 are lifters gripping the cable at 1/4, 1/2 and 3/4 span.
ROV 3 carries the connector. The mate point is the mid-span node -- the one
lifter 1 is stabilising -- because in a real operation you would stabilise
precisely where you need precision.

Everything is batched: one `step()` advances every environment at once.
"""

from typing import Optional, Tuple

import math
import torch

from .cable import Cable
from .config import (CABLE, ROV, SIM, SONAR, TASK, PROFILES, CableSpec,
                     CurrentProfile, ROVSpec, SimSpec, SonarSpec, TaskSpec)
from .rov import ROVSwarm
from .sonar import Sonar
from .water import CurrentField

PHASES = ("LIFT", "STABILISE", "MATE", "DONE")
LIFT, STABILISE, MATE, DONE = 0, 1, 2, 3
N_PHASE = len(PHASES)


class RepairEnv:
    """Batched seabed cable-repair environment.

    Observation is per-ROV and egocentric, so one shared policy serves every
    vehicle and the swarm size is not baked into the network.

    Action is [B, K, 4]: normalised surge/sway/heave/yaw wrench demand.
    """

    def __init__(
        self,
        batch: int = 64,
        profile: str | CurrentProfile = "calm",
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        cable_spec: CableSpec = CABLE,
        rov_spec: ROVSpec = ROV,
        sim_spec: SimSpec = SIM,
        task: TaskSpec = TASK,
        sonar_spec: SonarSpec = SONAR,
        use_sonar: bool = True,
    ) -> None:
        self.batch = batch
        self.device = torch.device(device)
        self.dtype = dtype
        self.cs, self.rs, self.sim, self.task = cable_spec, rov_spec, sim_spec, task
        self.use_sonar = use_sonar
        self.n_rov = task.n_rov

        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.profile = (PROFILES[profile] if isinstance(profile, str)
                        else profile)

        self.cable = Cable(cable_spec, batch, device=device, dtype=dtype)
        self.swarm = ROVSwarm(rov_spec, batch, self.n_rov, device=device,
                              dtype=dtype)
        self.field = CurrentField(self.profile, batch, device=device,
                                  dtype=dtype, generator=self.gen)
        # sonar targets per ROV: its own cable node, then each peer -> K per ROV
        self.sonar = Sonar(sonar_spec, batch, self.n_rov * self.n_rov,
                           device=device, dtype=dtype, generator=self.gen)

        N = cable_spec.n_nodes
        self.grip_nodes = torch.tensor(
            [N // 4, N // 2, (3 * N) // 4], device=self.device, dtype=torch.long)
        self.mate_node = int(N // 2)

        self.phase = torch.zeros(batch, device=self.device, dtype=torch.long)
        self.t = torch.zeros(batch, device=self.device, dtype=dtype)
        self.phase_t = torch.zeros(batch, device=self.device, dtype=dtype)
        self.mate_dwell = torch.zeros(batch, device=self.device, dtype=dtype)
        self.stable_dwell = torch.zeros(batch, device=self.device, dtype=dtype)
        self.last_action = torch.zeros(batch, self.n_rov, 4, device=self.device,
                                       dtype=dtype)
        self.done = torch.zeros(batch, device=self.device, dtype=torch.bool)
        self.success = torch.zeros(batch, device=self.device, dtype=torch.bool)
        self.fail_reason = torch.zeros(batch, device=self.device,
                                       dtype=torch.long)  # 0 none 1 tension
                                                           # 2 bend 3 timeout
                                                           # 4 grip lost
        self.lift_target_z = torch.zeros(batch, device=self.device, dtype=dtype)
        self.home_xy = torch.zeros(batch, self.n_rov, 2, device=self.device,
                                   dtype=dtype)

        # running episode statistics, read by the evaluator
        self.stat_peak_tension = torch.zeros(batch, device=self.device, dtype=dtype)
        self.stat_min_radius = torch.full((batch,), 1e6, device=self.device,
                                          dtype=dtype)
        self.stat_over_tension = torch.zeros(batch, device=self.device, dtype=dtype)
        self.stat_bend_events = torch.zeros(batch, device=self.device, dtype=dtype)
        self.stat_station_sq = torch.zeros(batch, device=self.device, dtype=dtype)
        self.stat_steps = torch.zeros(batch, device=self.device, dtype=dtype)
        self.stat_energy = torch.zeros(batch, device=self.device, dtype=dtype)

        self.obs_dim = self._probe_obs_dim()

    # ------------------------------------------------------------------ setup

    def _probe_obs_dim(self) -> int:
        self.reset()
        return int(self.observe().shape[-1])

    def set_profile(self, profile: str | CurrentProfile) -> None:
        """Switch sea state. Used by the curriculum between stages."""
        self.profile = PROFILES[profile] if isinstance(profile, str) else profile
        self.field = CurrentField(self.profile, self.batch, device=self.device,
                                  dtype=self.dtype, generator=self.gen)

    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset all environments, or only those selected by `mask` [B]."""
        if mask is None:
            mask = torch.ones(self.batch, device=self.device, dtype=torch.bool)
        if not bool(mask.any()):
            return self.observe()

        heading = (torch.rand(self.batch, generator=self.gen,
                              device=self.device, dtype=self.dtype)
                   * 2.0 * math.pi)
        self.cable.reset_seabed(heading=heading, mask=mask)
        self.cable._reset_viv(mask=mask)
        self.field.reset(mask)
        self.sonar.reset(mask)

        # lifters start just above their grip nodes, already attached
        pos = self.swarm.pos.clone()
        for k in range(3):
            pos[:, k] = self.cable.x[:, self.grip_nodes[k]]
            pos[:, k, 2] += 0.8
        # the connector vehicle starts at a standoff from the mate node
        pos[:, 3] = self.cable.x[:, self.mate_node]
        pos[:, 3, 0] += 1.5
        pos[:, 3, 2] += 0.6
        self.swarm.reset(pos, mask=mask)
        for k in range(3):
            self.swarm.grip_node[:, k] = torch.where(
                mask, torch.full_like(self.swarm.grip_node[:, k],
                                      int(self.grip_nodes[k])),
                self.swarm.grip_node[:, k])

        z = torch.zeros(self.batch, device=self.device, dtype=self.dtype)
        self.phase = torch.where(mask, torch.zeros_like(self.phase), self.phase)
        self.t = torch.where(mask, z, self.t)
        self.phase_t = torch.where(mask, z, self.phase_t)
        self.mate_dwell = torch.where(mask, z, self.mate_dwell)
        self.stable_dwell = torch.where(mask, z, self.stable_dwell)
        self.done = self.done & ~mask
        self.success = self.success & ~mask
        self.fail_reason = torch.where(mask, torch.zeros_like(self.fail_reason),
                                       self.fail_reason)
        self.last_action = torch.where(mask.view(-1, 1, 1), 0.0 * self.last_action,
                                       self.last_action)
        self.lift_target_z = torch.where(mask, z, self.lift_target_z)
        self.home_xy = torch.where(mask.view(-1, 1, 1), pos[..., :2], self.home_xy)

        for name in ("stat_peak_tension", "stat_over_tension",
                     "stat_bend_events", "stat_station_sq", "stat_steps",
                     "stat_energy"):
            setattr(self, name, torch.where(mask, z, getattr(self, name)))
        self.stat_min_radius = torch.where(mask, torch.full_like(z, 1e6),
                                           self.stat_min_radius)
        return self.observe()

    # ------------------------------------------------------------- geometry

    def _mate_frame(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mate node position, velocity and local cable heading."""
        n = self.mate_node
        p = self.cable.x[:, n]
        v = self.cable.v[:, n]
        seg = self.cable.x[:, n + 1] - self.cable.x[:, n - 1]
        heading = torch.atan2(seg[:, 1], seg[:, 0])
        return p, v, heading

    def connector_tip(self) -> torch.Tensor:
        """Where the connector actually is: 0.8 m up on a manipulator reach."""
        tip = self.swarm.pos[:, 3].clone()
        tip[:, 2] += 0.8
        return tip

    def targets(self) -> torch.Tensor:
        """[B, K, 3] commanded position for each ROV in the current phase."""
        tgt = self.swarm.pos.clone()
        mate_p, _, _ = self._mate_frame()

        # lifters: hold their home xy, with z ramping to the lift height
        ramp = (self.task.lift_rate * self.phase_t).clamp(
            max=self.task.lift_height)
        lift_z = torch.where(self.phase == LIFT, ramp,
                             torch.full_like(ramp, self.task.lift_height))
        for k in range(3):
            tgt[:, k, 0] = self.home_xy[:, k, 0]
            tgt[:, k, 1] = self.home_xy[:, k, 1]
            tgt[:, k, 2] = lift_z + 0.8

        # connector vehicle: stand off until MATE, then go for the node
        standoff = mate_p.clone()
        standoff[:, 0] += 1.5
        standoff[:, 2] += 0.6
        approach = mate_p.clone()
        approach[:, 2] -= 0.8                       # tip offset
        in_mate = (self.phase == MATE).unsqueeze(-1)
        tgt[:, 3] = torch.where(in_mate, approach, standoff)
        return tgt

    def mate_error(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Position error, relative speed and heading error of the connector."""
        mate_p, mate_v, heading = self._mate_frame()
        err = self.connector_tip() - mate_p
        rel_v = self.swarm.vel[:, 3] - mate_v
        dyaw = torch.remainder(self.swarm.yaw[:, 3] - heading + math.pi,
                               2.0 * math.pi) - math.pi
        return err, rel_v, dyaw

    # ---------------------------------------------------------- observation

    @staticmethod
    def to_body(vec: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        """World-frame vector -> each vehicle's body frame.

        Observations are egocentric for the same reason the actions are:
        thrusters act along the hull. Feeding world-frame vectors would make
        the policy learn a separate controller for every compass heading,
        when the control problem is identical in the body frame.
        """
        c, sn = torch.cos(yaw), torch.sin(yaw)
        bx = c * vec[..., 0] + sn * vec[..., 1]
        by = -sn * vec[..., 0] + c * vec[..., 1]
        return torch.stack([bx, by, vec[..., 2]], dim=-1)

    def observe(self) -> torch.Tensor:
        """[B, K, obs_dim] egocentric observation for each ROV."""
        B, K = self.batch, self.n_rov
        tgt = self.targets()
        d = self.cable.diagnostics()

        # --- things the vehicle must sense acoustically
        own_node = torch.zeros(B, K, 3, device=self.device, dtype=self.dtype)
        own_node_v = torch.zeros_like(own_node)
        for k in range(3):
            own_node[:, k] = self.cable.x[:, self.grip_nodes[k]] - self.swarm.pos[:, k]
            own_node_v[:, k] = self.cable.v[:, self.grip_nodes[k]] - self.swarm.vel[:, k]
        mate_p, mate_v, heading = self._mate_frame()
        own_node[:, 3] = mate_p - self.swarm.pos[:, 3]
        own_node_v[:, 3] = mate_v - self.swarm.vel[:, 3]

        peers = self.swarm.pos.unsqueeze(2) - self.swarm.pos.unsqueeze(1)
        peers_v = self.swarm.vel.unsqueeze(2) - self.swarm.vel.unsqueeze(1)
        # drop self-relation -> [B, K, K-1, 3]
        keep = ~torch.eye(K, dtype=torch.bool, device=self.device)
        peers = peers[:, keep].reshape(B, K, K - 1, 3)
        peers_v = peers_v[:, keep].reshape(B, K, K - 1, 3)

        if self.use_sonar:
            stack = torch.cat([own_node.unsqueeze(2), peers], dim=2)  # [B,K,K,3]
            flat = stack.reshape(B, K * K, 3)
            meas, valid = self.sonar.corrupt(flat)
            stack = meas.reshape(B, K, K, 3)
            own_node = stack[:, :, 0]
            peers = stack[:, :, 1:]
            vmask = valid.reshape(B, K, K)[:, :, 0].unsqueeze(-1).to(self.dtype)
        else:
            vmask = torch.ones(B, K, 1, device=self.device, dtype=self.dtype)

        # --- tension and curvature, felt locally through the gripper
        T = self.cable.tension / self.cs.working_tension
        radii = self.cable.bend_radii() / self.cs.min_bend_radius
        own_T = torch.zeros(B, K, 3, device=self.device, dtype=self.dtype)
        own_R = torch.zeros(B, K, 1, device=self.device, dtype=self.dtype)
        n_seg = self.cable.n_seg
        for k in range(3):
            j = int(self.grip_nodes[k])
            lo, hi = max(0, j - 2), min(n_seg, j + 1)
            own_T[:, k, 0] = T[:, min(j, n_seg - 1)]
            own_T[:, k, 1] = T[:, lo:hi].amax(dim=-1)
            own_T[:, k, 2] = T[:, lo:hi].amin(dim=-1)
            rlo, rhi = max(0, j - 3), min(radii.shape[-1], j + 2)
            own_R[:, k, 0] = radii[:, rlo:rhi].amin(dim=-1)
        own_T[:, 3] = T.amax(dim=-1, keepdim=True).expand(-1, 3)
        own_R[:, 3, 0] = radii.amin(dim=-1)

        # --- flow, estimated from the vehicle's own drag response (noisy)
        u = self.field.velocity(self.swarm.pos.reshape(B, -1, 3), self.t)
        u = u.reshape(B, K, 3)
        if self.use_sonar:
            u = u + torch.randn(u.shape, generator=self.gen, device=self.device,
                                dtype=self.dtype) * 0.03

        err = tgt - self.swarm.pos
        m_err, m_relv, m_dyaw = self.mate_error()
        role = torch.zeros(B, K, 2, device=self.device, dtype=self.dtype)
        role[:, :3, 0] = 1.0
        role[:, 3, 1] = 1.0
        phase_oh = torch.nn.functional.one_hot(self.phase, N_PHASE).to(self.dtype)
        phase_oh = phase_oh.unsqueeze(1).expand(B, K, N_PHASE)

        glob = torch.stack([
            d["tension_frac_working"].clamp(max=4.0),
            (d["min_bend_radius"] / self.cs.min_bend_radius).clamp(max=10.0),
            self.cable.x[..., 2].amax(dim=-1) / self.task.lift_height,
            self.phase_t / 20.0,
            self.mate_dwell / max(self.task.mate_dwell, 1e-6),
        ], dim=-1).unsqueeze(1).expand(B, K, 5)

        # mate error is what the connector vehicle acts on, so rotate it
        # into *its* body frame (ROV 3) before sharing it with the swarm
        yaw3 = self.swarm.yaw[:, 3:4]
        mate_block = torch.cat([
            self.to_body(m_err.unsqueeze(1), yaw3).squeeze(1),
            self.to_body(m_relv.unsqueeze(1), yaw3).squeeze(1),
            m_dyaw.unsqueeze(-1)], dim=-1).unsqueeze(1).expand(B, K, 7)

        yaw = self.swarm.yaw
        yaw_p = yaw.unsqueeze(-1).expand(B, K, K - 1)
        obs = torch.cat([
            self.to_body(err, yaw),                 # 3
            self.to_body(self.swarm.vel, yaw),      # 3
            torch.sin(self.swarm.yaw).unsqueeze(-1),
            torch.cos(self.swarm.yaw).unsqueeze(-1),
            self.swarm.yaw_rate.unsqueeze(-1),      # 3
            self.to_body(own_node, yaw),            # 3
            self.to_body(own_node_v, yaw),          # 3
            vmask,                                  # 1
            own_T, own_R,                           # 4
            self.to_body(peers, yaw_p).reshape(B, K, -1),    # 9
            self.to_body(peers_v, yaw_p).reshape(B, K, -1),  # 9
            self.to_body(u, yaw),                   # 3
            phase_oh,                               # 4
            self.last_action,                       # 4
            role,                                   # 2
            glob,                                   # 5
            mate_block,                             # 7
        ], dim=-1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------- step

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                  torch.Tensor, dict]:
        """One control step (3 physics steps at 150 Hz).

        Returns (obs, reward [B, K], done [B], info).
        """
        action = action.clamp(-1.0, 1.0)
        self.last_action = action
        dt = self.sim.dt
        alive = (~self.done).to(self.dtype)

        peak_T = torch.zeros(self.batch, device=self.device, dtype=self.dtype)
        min_R = torch.full((self.batch,), 1e6, device=self.device, dtype=self.dtype)

        for _ in range(self.sim.phys_per_control):
            f_cable, f_rov = self.swarm.gripper_forces(self.cable.x, self.cable.v)
            self.swarm.step(dt, action, self.field, self.t, grip_force=f_rov)
            self.cable.step(dt, self.field, self.t, ext_force=f_cable,
                            substeps=self.sim.xpbd_substeps,
                            max_velocity=self.sim.max_velocity)
            self.t = self.t + dt * alive
            d = self.cable.diagnostics()
            peak_T = torch.maximum(peak_T, d["peak_tension"])
            min_R = torch.minimum(min_R, d["min_bend_radius"])

        self.phase_t = self.phase_t + self.sim.control_dt * alive
        reward, info = self._reward_and_transitions(peak_T, min_R, action)
        return self.observe(), reward, self.done.clone(), info

    def _reward_and_transitions(self, peak_T: torch.Tensor, min_R: torch.Tensor,
                                action: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        task, cs = self.task, self.cs
        B, K = self.batch, self.n_rov
        alive = ~self.done

        tgt = self.targets()
        err = (tgt - self.swarm.pos).norm(dim=-1)                 # [B, K]
        lifter_err = err[:, :3].amax(dim=-1)
        m_err, m_relv, m_dyaw = self.mate_error()
        m_dist = m_err.norm(dim=-1)

        # ---- shaping terms
        t_frac = peak_T / cs.working_tension
        r_tension = -4.0 * (t_frac - 0.75).clamp_min(0.0).pow(2)
        r_bend = -4.0 * (1.0 - min_R / (1.6 * cs.min_bend_radius)).clamp_min(0.0)
        r_station = -1.2 * err.clamp(max=5.0)
        r_energy = -0.01 * action.abs().mean(dim=-1)

        # progress: lifters want height, the mater wants to close on the node
        height = self.cable.x[..., 2].amax(dim=-1)
        r_lift = torch.where(self.phase == LIFT,
                             0.8 * (height / task.lift_height).clamp(max=1.0),
                             torch.zeros_like(height))
        r_mate = torch.where(self.phase == MATE,
                             1.5 * (1.0 - (m_dist / 2.0).clamp(max=1.0)),
                             torch.zeros_like(m_dist))

        team = r_tension + r_bend + r_lift + r_mate
        reward = team.unsqueeze(-1).expand(B, K) + r_station + r_energy

        # ---- phase transitions
        lift_ok = (self.phase == LIFT) & (height > task.lift_height - 0.4) \
            & (lifter_err < 0.8)
        steady = (self.phase == STABILISE) & (lifter_err < task.station_tol)
        self.stable_dwell = torch.where(
            steady, self.stable_dwell + self.sim.control_dt,
            torch.where(self.phase == STABILISE,
                        torch.zeros_like(self.stable_dwell), self.stable_dwell))
        stab_ok = (self.phase == STABILISE) & (self.stable_dwell > 1.0)

        in_tol = (self.phase == MATE) & (m_dist < task.mate_pos_tol) \
            & (m_relv.norm(dim=-1) < 0.15) & (m_dyaw.abs() < task.mate_angle_tol)
        self.mate_dwell = torch.where(in_tol,
                                      self.mate_dwell + self.sim.control_dt,
                                      torch.zeros_like(self.mate_dwell))
        mate_ok = self.mate_dwell >= task.mate_dwell

        advance = (lift_ok | stab_ok) & alive
        self.phase = torch.where(advance, self.phase + 1, self.phase)
        self.phase_t = torch.where(advance, torch.zeros_like(self.phase_t),
                                   self.phase_t)
        reward = reward + (advance.to(self.dtype) * 8.0).unsqueeze(-1)

        # ---- terminations
        broke = peak_T > cs.break_tension
        bent = min_R < cs.min_bend_radius
        lost_grip = (self.swarm.grip_node[:, :3] < 0).any(dim=-1)
        timeout = torch.zeros_like(alive)
        for ph, limit in enumerate(task.phase_timeout):
            timeout = timeout | ((self.phase == ph) & (self.phase_t > limit))
        won = mate_ok & alive

        fail = (broke | bent | lost_grip | timeout) & alive & ~won
        reward = reward + (won.to(self.dtype) * 60.0).unsqueeze(-1)
        reward = reward - (fail.to(self.dtype) * 30.0).unsqueeze(-1)

        self.fail_reason = torch.where(
            fail & (self.fail_reason == 0),
            torch.where(broke, torch.ones_like(self.fail_reason),
                        torch.where(bent, 2 * torch.ones_like(self.fail_reason),
                                    torch.where(lost_grip,
                                                4 * torch.ones_like(self.fail_reason),
                                                3 * torch.ones_like(self.fail_reason)))),
            self.fail_reason)
        self.success = self.success | won
        self.phase = torch.where(won, torch.full_like(self.phase, DONE), self.phase)
        self.done = self.done | won | fail \
            | (self.t > self.sim.episode_seconds)

        # ---- episode statistics
        a = alive.to(self.dtype)
        self.stat_peak_tension = torch.maximum(self.stat_peak_tension, peak_T * a)
        self.stat_min_radius = torch.where(alive,
                                           torch.minimum(self.stat_min_radius, min_R),
                                           self.stat_min_radius)
        self.stat_over_tension += ((t_frac > 1.0) & alive).to(self.dtype)
        self.stat_bend_events += ((min_R < 1.25 * cs.min_bend_radius) & alive).to(self.dtype)
        self.stat_station_sq += lifter_err.pow(2) * a
        self.stat_steps += a
        self.stat_energy += action.abs().mean(dim=(-1, -2)) * a

        info = dict(phase=self.phase.clone(), success=self.success.clone(),
                    fail_reason=self.fail_reason.clone(),
                    peak_tension=peak_T, min_radius=min_R,
                    mate_dist=m_dist, lifter_err=lifter_err,
                    height=height)
        return reward, info

    # ---------------------------------------------------------- episode stats

    def episode_metrics(self) -> dict:
        """Per-environment summary, in the units the results table reports."""
        steps = self.stat_steps.clamp_min(1.0)
        return dict(
            success=self.success.to(self.dtype),
            peak_tension_frac=self.stat_peak_tension / self.cs.working_tension,
            peak_tension_n=self.stat_peak_tension,
            min_bend_radius=self.stat_min_radius,
            over_tension_events=self.stat_over_tension,
            bend_events=self.stat_bend_events,
            station_rms=(self.stat_station_sq / steps).sqrt(),
            time_s=steps * self.sim.control_dt,
            energy=self.stat_energy,
            fail_reason=self.fail_reason,
        )
