"""Degraded acoustic sensing.

There is no usable video at 3 km. Light scatters, the water is full of
sediment, and a work site with four thrusters running is a cloud of silt
within seconds. Real vehicles navigate on acoustics.

So the policy does not get ground truth. It gets range/bearing to the things
it cares about, with the error model of a real sonar:

  * range noise proportional to range (beam spreading)
  * bearing noise roughly constant in angle
  * dropout -- returns vanish entirely, in bursts, not independently
  * latency -- one control step of staleness, because acoustic processing is
    not free

This is what makes the autonomy problem non-trivial rather than a
coordinate-tracking exercise, and it is why the architecture needs a learned
policy: the right thruster command is a function of a noisy, intermittent
estimate of a state you cannot observe directly.

Dropout is modelled as a two-state Markov chain so losses come in bursts.
Independent per-step dropout would be far easier to filter and would flatter
the policy.
"""

from typing import Optional

import torch

from .config import SonarSpec


class Sonar:
    """Batched range/bearing sensor with bursty dropout.

    One instance serves a whole batch of environments and every tracked
    target within them: `corrupt` takes true relative positions
    [B, K, T, 3] and returns measured ones plus a validity mask.
    """

    def __init__(self, spec: SonarSpec, batch: int, n_obs: int,
                 device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32,
                 generator: Optional[torch.Generator] = None) -> None:
        self.spec = spec
        self.batch = batch
        self.n_obs = n_obs
        self.device = torch.device(device)
        self.dtype = dtype
        self._gen = generator
        # dropout state: True = currently dropped out
        self.dropped = torch.zeros(batch, n_obs, device=self.device,
                                   dtype=torch.bool)
        self.last_good = torch.zeros(batch, n_obs, 3, device=self.device,
                                     dtype=dtype)
        self.have_fix = torch.zeros(batch, n_obs, device=self.device,
                                    dtype=torch.bool)

    def reset(self, mask: Optional[torch.Tensor] = None) -> None:
        if mask is None:
            self.dropped.zero_()
            self.have_fix.zero_()
            self.last_good.zero_()
            return
        m = mask.to(self.device)
        self.dropped[m] = False
        self.have_fix[m] = False
        self.last_good[m] = 0.0

    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self._gen, device=self.device,
                          dtype=self.dtype)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self._gen, device=self.device,
                           dtype=self.dtype)

    def corrupt(self, rel: torch.Tensor) -> tuple:
        """True relative positions -> measured positions and a validity mask.

        Parameters
        ----------
        rel : [B, n_obs, 3] true target position relative to the sensor

        Returns
        -------
        measured : [B, n_obs, 3]  last good fix held through a dropout
        valid    : [B, n_obs]     False while dropped out
        """
        sp = self.spec
        B, K, _ = rel.shape

        # --- bursty dropout: two-state Markov chain
        p_enter = torch.full((B, K), sp.dropout_enter, device=self.device,
                             dtype=self.dtype)
        p_exit = torch.full((B, K), sp.dropout_exit, device=self.device,
                            dtype=self.dtype)
        roll = self._rand(B, K)
        enter = (~self.dropped) & (roll < p_enter)
        exit_ = self.dropped & (roll < p_exit)
        self.dropped = (self.dropped | enter) & ~exit_

        # --- range and bearing noise
        rng = rel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        unit = rel / rng
        rng_err = 1.0 + self._randn(B, K, 1) * sp.range_noise_frac
        # bearing error: a small random rotation, applied as a perturbation
        # perpendicular to the line of sight
        perp = self._randn(B, K, 3)
        perp = perp - (perp * unit).sum(-1, keepdim=True) * unit
        perp = perp / perp.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        bearing_err = self._randn(B, K, 1) * sp.bearing_noise_rad
        measured = (unit + perp * bearing_err)
        measured = measured / measured.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        measured = measured * rng * rng_err

        # --- hold the last good fix through a dropout. A real vehicle does
        #     this too; what it cannot do is know how stale the hold is.
        live = ~self.dropped
        upd = live.unsqueeze(-1)
        self.last_good = torch.where(upd, measured, self.last_good)
        self.have_fix = self.have_fix | live
        out = torch.where(self.have_fix.unsqueeze(-1), self.last_good, measured)
        return out, live
