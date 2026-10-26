"""Is the disturbance actually fast enough for loop latency to matter?

This is the test that decides whether the project's central claim is even
testable. The claim is that a 600 ms control loop cannot stabilise the cable.
That can only be true if the disturbance carries meaningful energy at
timescales near or below 600 ms (i.e. above ~1.7 Hz). If all the energy sits
at 10-second timescales, a delayed controller tracks it easily and there is
no result to report.

So we measure the power spectrum of the lateral force on a lifted cable span
and report how much of the disturbance energy lies above the latency corner
frequency. Run before trusting any teleop-vs-autonomy number.

Run: python -m tests.test_spectrum
"""

import sys

import torch

from sim.cable import Cable
from sim.config import CABLE, SIM, PROFILES
from sim.water import CurrentField

# Effective loop delay = observation delay + zero-order-hold half-period of
# the command rate + operator reaction lag. A pure delay L limits achievable
# closed-loop bandwidth to roughly omega = 1/L, i.e. f = 1/(2*pi*L): past
# that, feedback adds energy instead of removing it.
#
#   mode A: 600 obs + 250 ZOH (2 Hz) + 250 reaction = 1100 ms -> 0.145 Hz
#   mode B: 150 obs + 250 ZOH (2 Hz) + 250 reaction =  650 ms -> 0.245 Hz
#   onboard: 20 ms at 50 Hz, no network                       -> 7.96 Hz
import math

DELAYS_MS = {"teleop 600ms": 1100.0, "teleop 150ms": 650.0, "onboard": 20.0}
CORNERS = {k: 1.0 / (2.0 * math.pi * v / 1000.0) for k, v in DELAYS_MS.items()}


def force_trace(sea: str, seconds: float = 40.0, viv: bool = True,
                seed: int = 0) -> tuple:
    """Lateral force on the mid-span of a lifted cable, sampled at 120 Hz."""
    torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    spec = CABLE if viv else type(CABLE)(**{**CABLE.__dict__,
                                            "viv_enabled": False})
    cable = Cable(spec, 1, device="cpu")
    cable.reset_hanging(span=24.0, centre=(0.0, 0.0, 12.0))
    field = CurrentField(PROFILES[sea], 1, device="cpu", generator=gen)

    t = torch.zeros(1)
    trace = []
    steps = int(seconds * SIM.physics_hz)
    for _ in range(steps):
        cable.step(SIM.dt, field, t, substeps=SIM.xpbd_substeps,
                   max_velocity=SIM.max_velocity)
        t = t + SIM.dt
        # Full 3-component force on the central third of the span. All three
        # axes matter: vortex-shedding lift on a horizontal cable in a
        # horizontal flow acts *vertically*, so a horizontal-only probe sees
        # almost none of it.
        mid = cable._f_ext[0, cable.n_nodes // 3: 2 * cable.n_nodes // 3, :]
        trace.append(mid.sum(dim=0).clone())
    return torch.stack(trace), SIM.physics_hz


def spectrum_split(trace: torch.Tensor, fs: float, corner_hz: float) -> float:
    """Fraction of fluctuating force power above `corner_hz`, summed over axes."""
    x = trace - trace.mean(dim=0, keepdim=True)            # [T, 3]
    # Hann window to stop spectral leakage faking high-frequency content
    w = torch.hann_window(x.shape[0]).unsqueeze(-1)
    p = torch.fft.rfft(x * w, dim=0).abs().pow(2).sum(dim=-1)
    freq = torch.fft.rfftfreq(x.shape[0], d=1.0 / fs)
    return float(p[freq > corner_hz].sum() / p.sum().clamp_min(1e-30))


def main() -> int:
    print("Disturbance energy above the latency corner frequency")
    print("A pure delay L caps closed-loop bandwidth near f = 1/(2*pi*L).")
    print("Effective delay includes the command-rate hold and reaction lag.")
    print("corner frequencies: %s" % ", ".join(
        "%s = %.2f Hz" % (k, v) for k, v in CORNERS.items()))
    print("")
    print("%-9s %-5s %13s %13s %12s" % (
        "sea", "VIV", ">0.14Hz (A)", ">0.24Hz (B)", "rms force N"))
    print("-" * 56)

    rows = []
    for sea in ("moderate", "strong", "surge"):
        for viv in (False, True):
            tr, fs = force_trace(sea, seconds=25.0, viv=viv)
            f600 = spectrum_split(tr, fs, CORNERS["teleop 600ms"])
            f150 = spectrum_split(tr, fs, CORNERS["teleop 150ms"])
            rms = float((tr - tr.mean(dim=0, keepdim=True)).norm(dim=-1)
                        .pow(2).mean().sqrt())
            print("%-9s %-5s %12.1f%% %12.1f%% %12.1f" % (
                sea, "on" if viv else "off", 100 * f600, 100 * f150, rms))
            rows.append((sea, viv, f600, f150, rms))

    print()
    viv_on = [r for r in rows if r[1]]
    viv_off = [r for r in rows if not r[1]]
    gain = (sum(r[2] for r in viv_on) / len(viv_on)) / \
           max(sum(r[2] for r in viv_off) / len(viv_off), 1e-9)
    print("VIV multiplies above-corner energy share by %.2fx" % gain)

    worst = max(r[2] for r in viv_on)
    if worst < 0.10:
        print("\nVERDICT: disturbance is too slow. Only %.1f%% of energy sits "
              "above the 600 ms corner." % (100 * worst))
        print("A delayed controller would track this, and the latency claim "
              "would NOT be supportable.")
        return 1
    print("\nVERDICT: %.1f%% of disturbance energy lies above the 600 ms "
          "corner." % (100 * worst))
    print("Loop latency is measurable in this physics. The comparison is "
          "meaningful.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
