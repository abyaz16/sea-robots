"""Batched XPBD discrete-rod model of an armoured submarine cable.

Why XPBD
--------
An armoured cable is effectively inextensible: its axial stiffness is stiff
enough that an explicit force-based solver needs a timestep in the tens of
microseconds to stay stable. XPBD (Extended Position Based Dynamics, Macklin
et al. 2016) enforces inextensibility as a *constraint projection* instead,
so the stiff mode costs iterations rather than timestep, and the result is
stable at 120 Hz. It is also compliance-parametrised, so the constraint
stiffness is a physical number (1/alpha) rather than a tuning knob that
changes meaning with the timestep.

The whole solver is batched: positions are [B, N, 3] and one `step()` advances
every environment in the batch. Constraints are solved with graph colouring
(even/odd for stretch, mod-3 for bending) so each colour is a single fully
parallel tensor operation -- that is what makes thousands of parallel
environments feasible on one GPU, and therefore what makes PPO feasible in
days instead of weeks.

Modelled failure modes
----------------------
1. Over-tension  -- exceeding break tension destroys the cable.
2. Over-bending  -- violating minimum bend radius destroys the *fibre*
                    without any visible damage to the cable.

(2) is the interesting one. It means lifting harder is not simply safer:
lifting harder flattens the sag, which tightens the bend at the lift points.
The policy has to find the narrow band between too much sag and too sharp a
bend, while being pushed around by the current.
"""

from typing import Optional, Tuple

import math
import torch

from .config import CableSpec, RHO_WATER, GRAVITY
from .water import CurrentField


def _arc_init(span: float, arc_len: float) -> Tuple[float, float]:
    """Half-angle and radius of a circular arc with given chord and length.

    Used to initialise a hanging cable in a shape that already satisfies the
    inextensibility constraint. Initialising on the straight chord instead
    would start every segment compressed, and the first solve would fire the
    cable apart -- the single most common way a rod sim 'explodes'.
    """
    if arc_len <= span * (1.0 + 1e-9):
        return 0.0, float("inf")
    target = span / arc_len
    lo, hi = 1e-6, math.pi - 1e-6
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if math.sin(mid) / mid > target:
            lo = mid
        else:
            hi = mid
    alpha = 0.5 * (lo + hi)
    return alpha, arc_len / (2.0 * alpha)


class Cable:
    """A batch of B independent cables, each N nodes long.

    Attributes
    ----------
    x        : [B, N, 3] node positions
    v        : [B, N, 3] node velocities
    w        : [B, N]    inverse effective mass (0 => pinned / anchored)
    tension  : [B, N-1]  last-solved segment tension, N
    """

    def __init__(
        self,
        spec: CableSpec,
        batch: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.spec = spec
        self.batch = batch
        self.device = torch.device(device)
        self.dtype = dtype

        N = spec.n_nodes
        self.n_nodes = N
        self.n_seg = N - 1

        # Effective mass includes added mass -- the water a moving cable has
        # to shove aside. For this geometry it is ~57% of the dry mass, so
        # omitting it would make the cable respond far too briskly.
        self.m_dry = spec.node_mass
        self.m_added = spec.ca_normal * RHO_WATER * spec.node_volume
        self.m_eff = self.m_dry + self.m_added

        # Net submerged weight per node, N (downward)
        self.w_sub = (spec.node_mass - RHO_WATER * spec.node_volume) * GRAVITY

        self.x = torch.zeros(batch, N, 3, device=self.device, dtype=dtype)
        self.v = torch.zeros_like(self.x)
        self.w = torch.full((batch, N), 1.0 / self.m_eff,
                            device=self.device, dtype=dtype)
        self.tension = torch.zeros(batch, self.n_seg,
                                   device=self.device, dtype=dtype)

        # --- constraint colourings, precomputed once
        seg = torch.arange(self.n_seg, device=self.device)
        self._stretch_colours = [seg[seg % 2 == 0], seg[seg % 2 == 1]]
        joint = torch.arange(1, N - 1, device=self.device)
        self._bend_colours = [joint[joint % 3 == c] for c in range(3)]

        # scratch for drag, refreshed once per physics step
        self._f_ext = torch.zeros_like(self.x)

        # VIV shedding phase, integrated per node because the shedding
        # frequency follows the *local* cross-flow speed.
        self.viv_phase = torch.zeros(batch, N, device=self.device, dtype=dtype)
        self.viv_offset = torch.zeros(batch, N, device=self.device, dtype=dtype)
        self._reset_viv()

    def _reset_viv(self, mask: Optional[torch.Tensor] = None) -> None:
        """Draw a spanwise-correlated random phase field for vortex shedding.

        Shedding is not coherent along a long cylinder: it organises into
        cells of a few diameters. We model that as a smooth random phase
        along the arc, with correlation length `viv_corr_length`. Fully
        coherent shedding would overstate the net load on the span by a large
        factor; fully random would understate it.
        """
        sp = self.spec
        N = self.n_nodes
        new_phase = torch.empty_like(self.viv_phase).uniform_(
            0.0, 2.0 * math.pi)
        self.viv_phase = (new_phase if mask is None else
                          torch.where(mask.view(-1, 1), new_phase, self.viv_phase))
        # smooth noise along the span: a few low-order harmonics whose
        # wavelength is set by the correlation length
        span = (N - 1) * sp.seg_len
        n_harm = max(1, int(span / max(sp.viv_corr_length, 1e-6)))
        s = torch.arange(N, device=self.device, dtype=self.dtype) * sp.seg_len
        off = torch.zeros(self.batch, N, device=self.device, dtype=self.dtype)
        for _ in range(3):
            k = (2.0 * math.pi / span) * float(
                torch.randint(1, n_harm + 1, (1,)).item())
            ph = torch.rand(self.batch, 1, device=self.device,
                            dtype=self.dtype) * (2.0 * math.pi)
            off = off + torch.sin(k * s.unsqueeze(0) + ph)
        new_off = off * (math.pi / 3.0)
        self.viv_offset = (new_off if mask is None else
                           torch.where(mask.view(-1, 1), new_off, self.viv_offset))

    # ------------------------------------------------------------------ setup

    def pin(self, node_idx: int, value: bool = True) -> None:
        """Anchor (or release) a node across the whole batch."""
        self.w[:, node_idx] = 0.0 if value else 1.0 / self.m_eff

    def reset_hanging(self, span: float, centre: Tuple[float, float, float],
                      jitter: float = 0.0) -> None:
        """Both ends pinned, cable hanging as a circular arc between them.

        This is the Day-1 stability fixture: the hardest static configuration
        for a rod solver, because every segment carries real tension and the
        sag is deep.
        """
        N, L = self.n_nodes, self.spec.seg_len
        arc_len = self.n_seg * L
        alpha, R = _arc_init(span, arc_len)

        if math.isinf(R):
            s = torch.linspace(-0.5, 0.5, N, device=self.device, dtype=self.dtype)
            px = s * span
            pz = torch.zeros_like(px)
        else:
            th = torch.linspace(-alpha, alpha, N, device=self.device,
                                dtype=self.dtype)
            px = R * torch.sin(th)
            pz = -R * (torch.cos(th) - math.cos(alpha))   # 0 at the ends

        cx, cy, cz = centre
        self.x[:] = torch.stack(
            [px + cx, torch.full_like(px, cy), pz + cz], dim=-1).unsqueeze(0)
        if jitter > 0.0:
            self.x += torch.randn_like(self.x) * jitter
        self.v.zero_()
        self.tension.zero_()
        self.w[:] = 1.0 / self.m_eff
        self.pin(0)
        self.pin(N - 1)

    def reset_seabed(self, heading=0.0,
                     centre: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                     sag_into_bed: float = 0.02,
                     mask: Optional[torch.Tensor] = None) -> None:
        """Cable lying slack on the seabed -- the pre-repair state.

        `heading` may be a float or a [B] tensor, and `mask` restricts the
        reset to selected environments. Partial reset matters: a vectorised
        RL loop finishes episodes at different times, and resetting the whole
        batch whenever one environment ends would silently truncate the rest.
        """
        N, L = self.n_nodes, self.spec.seg_len
        s = (torch.arange(N, device=self.device, dtype=self.dtype)
             - 0.5 * (N - 1)) * L
        if not torch.is_tensor(heading):
            heading = torch.full((self.batch,), float(heading),
                                 device=self.device, dtype=self.dtype)
        c = torch.cos(heading).unsqueeze(-1)
        sn = torch.sin(heading).unsqueeze(-1)
        x_new = torch.stack([
            s.unsqueeze(0) * c + centre[0],
            s.unsqueeze(0) * sn + centre[1],
            torch.full((self.batch, N), centre[2] + sag_into_bed,
                       device=self.device, dtype=self.dtype),
        ], dim=-1)

        if mask is None:
            self.x = x_new
            self.v.zero_()
            self.tension.zero_()
            self.w[:] = 1.0 / self.m_eff
            return
        m3 = mask.view(-1, 1, 1)
        self.x = torch.where(m3, x_new, self.x)
        self.v = torch.where(m3, torch.zeros_like(self.v), self.v)
        self.tension = torch.where(mask.view(-1, 1),
                                   torch.zeros_like(self.tension), self.tension)
        self.w = torch.where(mask.view(-1, 1),
                             torch.full_like(self.w, 1.0 / self.m_eff), self.w)

    # ------------------------------------------------------------- geometry

    def segments(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, N-1, 3] vectors from node i to node i+1."""
        x = self.x if x is None else x
        return x[:, 1:] - x[:, :-1]

    def tangents(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, N, 3] unit tangent at each node (averaged over adjoining segs)."""
        d = self.segments(x)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        t = torch.zeros_like(self.x if x is None else x)
        t[:, :-1] += d
        t[:, 1:] += d
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-9)

    def turn_angles(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, N-2] turning angle at each interior joint, radians."""
        d = self.segments(x)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        cos = (d[:, :-1] * d[:, 1:]).sum(-1).clamp(-1.0, 1.0)
        return torch.acos(cos)

    def bend_radii(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """[B, N-2] local radius of curvature at each interior joint, m.

        Inverts phi = 2*asin(L / 2R), the exact relation for a discrete rod
        following a circular arc. Returns a large finite value when straight.
        """
        phi = self.turn_angles(x)
        s = torch.sin(0.5 * phi).clamp_min(1e-6)
        return (0.5 * self.spec.seg_len / s).clamp(max=1.0e6)

    # ------------------------------------------------------------------ loads

    def hydro_forces(self, current: CurrentField, t: torch.Tensor,
                     dt: float) -> torch.Tensor:
        """[B, N, 3] gravity + buoyancy + directional drag on every node.

        Drag uses the Morison cross-flow/axial split: a cable broadside to the
        flow sees ~100x the drag it sees end-on. That asymmetry is real and
        exploitable -- the ROVs can reduce load by how they orient the span.
        """
        sp = self.spec
        u = current.velocity(self.x, t)
        v_rel = self.v - u
        tang = self.tangents()

        v_t = (v_rel * tang).sum(-1, keepdim=True) * tang
        v_n = v_rel - v_t
        area_n = sp.diameter * sp.seg_len
        area_t = math.pi * sp.diameter * sp.seg_len
        f = -0.5 * RHO_WATER * sp.cd_normal * area_n \
            * v_n.norm(dim=-1, keepdim=True) * v_n
        f = f - 0.5 * RHO_WATER * sp.cd_tangent * area_t \
            * v_t.norm(dim=-1, keepdim=True) * v_t

        f[..., 2] -= self.w_sub          # net submerged weight

        if sp.viv_enabled:
            f = f + self._viv_force(v_n, tang, dt)
        return f

    def _viv_force(self, v_n: torch.Tensor, tang: torch.Tensor,
                   dt: float) -> torch.Tensor:
        """Cross-flow vortex-shedding lift, [B, N, 3].

        Lift acts perpendicular to both the cylinder axis and the local
        cross-flow, oscillating at the Strouhal frequency f = St*|v_n|/D.
        The phase is integrated rather than evaluated from absolute time,
        because the local frequency changes as the flow does -- evaluating
        sin(2*pi*f(t)*t) instead would produce a spurious chirp.
        """
        sp = self.spec
        speed_n = v_n.norm(dim=-1)                                # [B, N]
        f_shed = sp.viv_strouhal * speed_n / sp.diameter
        self.viv_phase = torch.remainder(
            self.viv_phase + 2.0 * math.pi * f_shed * dt, 2.0 * math.pi)

        lift_dir = torch.cross(tang, v_n, dim=-1)
        lift_dir = lift_dir / lift_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        area_n = sp.diameter * sp.seg_len
        mag = (0.5 * RHO_WATER * sp.viv_cl * area_n * speed_n.pow(2)
               * torch.sin(self.viv_phase + self.viv_offset))
        return mag.unsqueeze(-1) * lift_dir

    # ------------------------------------------------------------------- step

    def step(self, dt: float, current: CurrentField, t: torch.Tensor,
             ext_force: Optional[torch.Tensor] = None,
             substeps: int = 8, max_velocity: float = 25.0,
             seabed_z: float = 0.0, seabed_friction: float = 0.5) -> None:
        """Advance the cable by `dt` using `substeps` XPBD substeps.

        Hydrodynamic loads are evaluated once per `step` and held across the
        substeps. That is the standard XPBD arrangement: constraint projection
        is what needs the small substep, while the smooth external load does
        not, and re-sampling the current field eight times per step would cost
        more than it buys.
        """
        sub_dt = dt / substeps
        f = self.hydro_forces(current, t, dt)
        if ext_force is not None:
            f = f + ext_force
        self._f_ext = f
        acc = f * self.w.unsqueeze(-1)        # w = 0 pins a node automatically

        for _ in range(substeps):
            self.v = self.v + sub_dt * acc
            self.v = self.v.clamp(-max_velocity, max_velocity)
            x_prev = self.x
            x = self.x + sub_dt * self.v

            x, lam = self._solve_constraints(x.clone(), sub_dt)
            x = self._seabed_contact(x, x_prev, seabed_z, seabed_friction)

            self.v = (x - x_prev) / sub_dt
            self.x = x
            self.tension = lam / (sub_dt * sub_dt)

    # -------------------------------------------------------- constraint solve

    def _solve_constraints(self, x: torch.Tensor, h: float,
                           iters: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        sp = self.spec
        L = sp.seg_len
        a_stretch = sp.axial_compliance / (h * h)
        a_bend = sp.bend_compliance / (h * h)

        lam_stretch = torch.zeros(self.batch, self.n_seg,
                                  device=self.device, dtype=self.dtype)
        lam_bend = torch.zeros(self.batch, self.n_nodes - 2,
                               device=self.device, dtype=self.dtype)

        for _ in range(iters):
            # --- inextensibility: |x_{j+1} - x_j| = L
            for idx in self._stretch_colours:
                if idx.numel() == 0:
                    continue
                i0, i1 = idx, idx + 1
                p0, p1 = x[:, i0], x[:, i1]
                d = p1 - p0
                dist = d.norm(dim=-1).clamp_min(1e-9)
                n = d / dist.unsqueeze(-1)
                C = dist - L
                w0, w1 = self.w[:, i0], self.w[:, i1]
                wsum = w0 + w1
                lam = lam_stretch[:, idx]
                dlam = (-C - a_stretch * lam) / (wsum + a_stretch).clamp_min(1e-12)
                lam_stretch[:, idx] = lam + dlam
                dp = dlam.unsqueeze(-1) * n
                # grad_{p0} C = -n, grad_{p1} C = +n
                x.index_add_(1, i0, -w0.unsqueeze(-1) * dp)
                x.index_add_(1, i1, w1.unsqueeze(-1) * dp)

            # --- bending: resist closing the (j-1, j+1) gap below 2L.
            #     d = 2L cos(phi/2), so this is a monotone, one-sided
            #     penalty on turn angle: cheap, symmetric, and stable, where
            #     an explicit angle gradient is neither.
            for idx in self._bend_colours:
                if idx.numel() == 0:
                    continue
                im, ip = idx - 1, idx + 1
                pm, pp = x[:, im], x[:, ip]
                d = pp - pm
                dist = d.norm(dim=-1).clamp_min(1e-9)
                n = d / dist.unsqueeze(-1)
                C = dist - 2.0 * L
                active = (C < 0.0).to(self.dtype)      # only resist bending
                wm, wp = self.w[:, im], self.w[:, ip]
                wsum = wm + wp
                lam = lam_bend[:, idx - 1]
                dlam = (-C - a_bend * lam) / (wsum + a_bend).clamp_min(1e-12)
                dlam = dlam * active
                lam_bend[:, idx - 1] = lam + dlam
                dp = dlam.unsqueeze(-1) * n
                x.index_add_(1, im, -wm.unsqueeze(-1) * dp)
                x.index_add_(1, ip, wp.unsqueeze(-1) * dp)

        return x, lam_stretch

    # ------------------------------------------------------------- seabed

    def _seabed_contact(self, x: torch.Tensor, x_prev: torch.Tensor,
                        z0: float, mu: float) -> torch.Tensor:
        """Non-penetration plus Coulomb friction against the mud.

        Reduced-order: real soil adds burial suction and strain-rate effects
        that can dominate the de-burial load. Noted in the limits section.
        """
        pen = (z0 - x[..., 2]).clamp_min(0.0)
        if not bool((pen > 0).any()):
            return x
        x = x.clone()
        x[..., 2] = x[..., 2] + pen

        # Tangential friction, applied as a positional correction bounded by
        # mu * normal penetration depth -- the standard PBD friction model.
        dxy = x[..., :2] - x_prev[..., :2]
        slide = dxy.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        limit = (mu * pen).unsqueeze(-1)
        corr = torch.minimum(slide, limit) * (dxy / slide)
        contact = (pen > 0).unsqueeze(-1).to(self.dtype)
        x[..., :2] = x[..., :2] - corr * contact
        return x

    # -------------------------------------------------------- diagnostics

    def diagnostics(self) -> dict:
        """Scalars per environment, for reward shaping, logging and eval."""
        sp = self.spec
        seg_len = self.segments().norm(dim=-1)
        radii = self.bend_radii()
        peak_t = self.tension.abs().amax(dim=-1)
        min_r = radii.amin(dim=-1)
        return dict(
            peak_tension=peak_t,
            tension_frac_break=peak_t / sp.break_tension,
            min_bend_radius=min_r,
            bend_violation=(min_r < sp.min_bend_radius),
            broke=(peak_t > sp.break_tension),
            strain_max=(seg_len / sp.seg_len - 1.0).abs().amax(dim=-1),
            height_max=self.x[..., 2].amax(dim=-1),
            speed_max=self.v.norm(dim=-1).amax(dim=-1),
            over_working=(peak_t > sp.working_tension),
            tension_frac_working=peak_t / sp.working_tension,
            finite=torch.isfinite(self.x).flatten(1).all(dim=1),
        )
