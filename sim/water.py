"""Ocean current field and hydrodynamic loads, batched in PyTorch.

Design notes
------------
The current field is deliberately *reduced-order*: it is not a CFD solve. It
is the superposition of three things that matter for station-keeping control:

  1. a sheared mean flow        -- the steady load the ROV must trim against
  2. band-limited turbulence    -- the broadband disturbance it must reject
  3. periodic surge gusts       -- the transient that defeats a laggy loop

The turbulence is built from a small sum of Fourier modes with each mode's
velocity direction perpendicular to its wavevector, which makes the
fluctuating field divergence-free (incompressible) by construction. That is
cheap, smooth in space and time, exactly reproducible from a seed, and
differentiable -- all properties a CFD field would not give us inside an RL
training loop running thousands of environments in parallel.

Everything is shaped [B, ...] so one call evaluates the field for every
environment in the batch at once.
"""

from typing import Optional

import math
import torch

from .config import RHO_WATER, CurrentProfile

N_TURB_MODES = 6


class CurrentField:
    """Batched, seeded, analytic current field.

    Parameters
    ----------
    profile : CurrentProfile
        The sea state to realise.
    batch : int
        Number of independent environments. Each gets its own flow heading,
        turbulence phases and gust timing, so a batch is a batch of genuinely
        different conditions drawn from the same distribution.
    """

    def __init__(
        self,
        profile: CurrentProfile,
        batch: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        self.profile = profile
        self.batch = batch
        self.device = torch.device(device)
        self.dtype = dtype
        self._gen = generator
        self.reset()

    # ------------------------------------------------------------------ utils

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self._gen, device=self.device,
                          dtype=self.dtype)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self._gen, device=self.device,
                           dtype=self.dtype)

    # ------------------------------------------------------------------ reset

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        """Draw a new realisation of the field.

        `mask` is a bool tensor [B]; only those environments are re-drawn,
        which is what a vectorised env needs on partial episode resets.
        """
        p = self.profile
        B = self.batch
        new = self._draw(B)
        if mask is None or not hasattr(self, "heading"):
            for k, v in new.items():
                setattr(self, k, v)
            return
        m = mask.to(self.device)
        for k, v in new.items():
            cur = getattr(self, k)
            sel = m.view(-1, *([1] * (cur.dim() - 1))).expand_as(cur)
            setattr(self, k, torch.where(sel, v, cur))

    def _draw(self, B: int) -> dict:
        p = self.profile
        theta = self._rand(B) * (2.0 * math.pi)
        flow_dir = torch.stack(
            [torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=-1)

        # Turbulence modes. Wavelengths 2-40 m, periods 3-25 s: the band that
        # actually moves a 1.5 t vehicle. Shorter scales average out over the
        # hull, longer ones are indistinguishable from the mean flow.
        wavelen = 2.0 + self._rand(B, N_TURB_MODES) * 38.0
        kdir = self._randn(B, N_TURB_MODES, 3)
        kdir = kdir / kdir.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        kvec = kdir * (2.0 * math.pi / wavelen).unsqueeze(-1)

        # Velocity direction perpendicular to k => divergence-free field.
        aux = self._randn(B, N_TURB_MODES, 3)
        udir = torch.cross(kdir, aux, dim=-1)
        udir = udir / udir.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        omega = (2.0 * math.pi) / (3.0 + self._rand(B, N_TURB_MODES) * 22.0)
        phase = self._rand(B, N_TURB_MODES) * (2.0 * math.pi)
        # 1/k amplitude roll-off: an energy cascade, crudely. Normalised so
        # the summed RMS matches the requested turbulence intensity.
        amp = 1.0 / (1.0 + kvec.norm(dim=-1))
        amp = amp / amp.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-9)
        amp = amp * (p.turb_intensity * p.u_ref) * math.sqrt(2.0)

        lo, hi = p.gust_period
        gust_period = lo + self._rand(B) * (hi - lo)
        gust_phase = self._rand(B) * gust_period
        gtheta = theta + (self._randn(B) * 0.35)   # gust roughly follows flow
        gust_dir = torch.stack(
            [torch.cos(gtheta), torch.sin(gtheta),
             self._randn(B) * 0.15], dim=-1)
        gust_dir = gust_dir / gust_dir.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        return dict(
            heading=theta, flow_dir=flow_dir,
            kvec=kvec, udir=udir, omega=omega, phase=phase, amp=amp,
            gust_period=gust_period, gust_phase=gust_phase, gust_dir=gust_dir,
        )

    # ------------------------------------------------------------------ field

    def gust_envelope(self, t: torch.Tensor) -> torch.Tensor:
        """Gust strength in [0, gust_amp] at time `t` [B]. Gaussian in time.

        Returns zero everywhere for profiles with gust_amp == 0.
        """
        p = self.profile
        if p.gust_amp <= 0.0:
            return torch.zeros_like(t)
        # Signed distance to the nearest gust peak, wrapped into the period.
        tau = torch.remainder(t - self.gust_phase, self.gust_period)
        tau = torch.minimum(tau, self.gust_period - tau)
        return p.gust_amp * torch.exp(-0.5 * (tau / p.gust_width) ** 2)

    def mean_speed(self, z: torch.Tensor) -> torch.Tensor:
        """Sheared mean speed at height `z` above the seabed.

        A 1/7-power boundary-layer profile, the standard engineering fit for
        a rough-wall benthic boundary layer. Clamped near the bed so the
        gradient stays finite.
        """
        p = self.profile
        zz = z.clamp_min(0.05)
        return p.u_ref * (zz / p.z_ref).pow(1.0 / 7.0).clamp(max=1.6)

    def velocity(self, pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Water velocity at world positions `pos`.

        Parameters
        ----------
        pos : [B, K, 3] world positions (z measured up from the seabed)
        t   : [B] simulation time, seconds

        Returns
        -------
        [B, K, 3] water velocity, m/s
        """
        B, K, _ = pos.shape
        z = pos[..., 2]                                      # [B, K]

        # --- mean shear flow
        u = self.mean_speed(z).unsqueeze(-1) * self.flow_dir.unsqueeze(1)

        # --- turbulence: sum_m amp_m * udir_m * sin(k_m . x - w_m t + phi_m)
        kx = torch.einsum("bkd,bmd->bkm", pos, self.kvec)
        arg = kx - (self.omega * t.unsqueeze(-1)).unsqueeze(1) \
                 + self.phase.unsqueeze(1)
        s = torch.sin(arg) * self.amp.unsqueeze(1)           # [B, K, M]
        u = u + torch.einsum("bkm,bmd->bkd", s, self.udir)

        # --- surge gust: coherent across the work site, so it loads every
        #     vehicle and the whole cable span at the same instant. That
        #     coherence is what makes it a control problem and not just noise.
        g = self.gust_envelope(t) * self.profile.u_ref        # [B]
        u = u + (g.view(B, 1, 1) * self.gust_dir.unsqueeze(1))
        return u


# --------------------------------------------------------------- hydro loads

def morison_cylinder(
    v_rel: torch.Tensor,
    tangent: torch.Tensor,
    diameter: float,
    length: float,
    cd_normal: float,
    cd_tangent: float,
) -> torch.Tensor:
    """Drag on a slender cylinder, split into cross-flow and axial parts.

    This is the drag half of the Morison equation. For a cable, the
    directional split matters a lot: a 30 m span broadside to the flow sees
    roughly 100x the drag it sees end-on, and that asymmetry is precisely
    what the ROVs can exploit by choosing how to orient the lifted span.

    Parameters
    ----------
    v_rel   : [..., 3] velocity of the body relative to the water
    tangent : [..., 3] unit vector along the cylinder axis
    """
    v_t = (v_rel * tangent).sum(-1, keepdim=True) * tangent
    v_n = v_rel - v_t
    area_n = diameter * length
    area_t = math.pi * diameter * length          # wetted area for skin drag
    f_n = -0.5 * RHO_WATER * cd_normal * area_n * v_n.norm(dim=-1, keepdim=True) * v_n
    f_t = -0.5 * RHO_WATER * cd_tangent * area_t * v_t.norm(dim=-1, keepdim=True) * v_t
    return f_n + f_t


def quadratic_drag(v_rel: torch.Tensor, cd: torch.Tensor | float,
                   area: torch.Tensor | float) -> torch.Tensor:
    """Bluff-body drag, F = -0.5 rho Cd A |v| v, applied per axis."""
    speed = v_rel.norm(dim=-1, keepdim=True)
    return -0.5 * RHO_WATER * cd * area * speed * v_rel
