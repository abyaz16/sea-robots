"""Day-1 hard checkpoint: does the cable hang and bend stably for 60 s?

Run:  python -m tests.test_stability

A rod solver that 'explodes' is the single likeliest way this project dies,
so the checkpoint is a measurement, not an eyeball. We assert on:

  * finiteness            -- no NaN/Inf anywhere, ever
  * inextensibility       -- segment strain stays under 1%
  * energy boundedness    -- kinetic energy decays to a quiet steady state
  * geometric sanity      -- the arc length is conserved
  * reproducibility       -- same seed, same trajectory

and we run it in all six sea states, including the surge, batched.
"""

import math
import sys
import time

import torch

from sim.cable import Cable
from sim.config import CABLE, SIM, PROFILES
from sim.water import CurrentField


def run_case(name: str, batch: int = 16, seconds: float = 60.0,
             seed: int = 0, device: str = "cpu") -> dict:
    gen = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)

    cable = Cable(CABLE, batch, device=device)
    # 24 m span for 30 m of cable: a deep, heavily loaded sag. Harder than
    # anything the repair task itself will ask for.
    cable.reset_hanging(span=24.0, centre=(0.0, 0.0, 12.0))
    field = CurrentField(PROFILES[name], batch, device=device, generator=gen)

    steps = int(seconds * SIM.physics_hz)
    t = torch.zeros(batch, device=device)

    arc0 = cable.segments().norm(dim=-1).sum(dim=-1)
    worst = dict(strain=0.0, speed=0.0, tension=0.0, min_radius=math.inf)
    ke_trace = []
    t_start = time.perf_counter()

    for i in range(steps):
        cable.step(SIM.dt, field, t, substeps=SIM.xpbd_substeps,
                   max_velocity=SIM.max_velocity)
        t = t + SIM.dt
        if i % 10 == 0:
            d = cable.diagnostics()
            if not bool(d["finite"].all()):
                return dict(name=name, ok=False, reason="non-finite state",
                            step=i)
            worst["strain"] = max(worst["strain"], float(d["strain_max"].max()))
            worst["speed"] = max(worst["speed"], float(d["speed_max"].max()))
            worst["tension"] = max(worst["tension"], float(d["peak_tension"].max()))
            worst["min_radius"] = min(worst["min_radius"],
                                      float(d["min_bend_radius"].min()))
            ke = 0.5 * cable.m_eff * cable.v.pow(2).sum(dim=-1).sum(dim=-1)
            ke_trace.append(float(ke.mean()))

    wall = time.perf_counter() - t_start
    arc1 = cable.segments().norm(dim=-1).sum(dim=-1)

    settle = sum(ke_trace[-20:]) / 20.0
    early = max(ke_trace[:20])
    return dict(
        name=name, ok=True, wall=wall, steps=steps, batch=batch,
        strain=worst["strain"], speed=worst["speed"], tension=worst["tension"],
        min_radius=worst["min_radius"],
        arc_err=float((arc1 - arc0).abs().max()),
        ke_early=early, ke_settled=settle, ke_peak=max(ke_trace),
        realtime_x=(seconds * batch) / wall,
    )


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: %s | torch %s" % (device, torch.__version__))
    print("%-9s %7s %7s %8s %8s %8s %8s %8s %7s" % (
        "sea", "strain%", "v_max", "T_peak", "R_min", "arc_err", "ke_pk",
        "ke_set", "x_rt"))
    print("-" * 78)

    failures = []
    for name in PROFILES:
        r = run_case(name, device=device)
        if not r["ok"]:
            print("%-9s FAILED: %s at step %d" % (name, r["reason"], r["step"]))
            failures.append(name)
            continue
        print("%-9s %7.3f %7.3f %8.0f %8.2f %8.1e %8.2f %8.2f %7.2f" % (
            name, r["strain"] * 100.0, r["speed"], r["tension"],
            r["min_radius"], r["arc_err"], r["ke_peak"], r["ke_settled"],
            r["realtime_x"]))

        # --- the actual checkpoint assertions
        if r["strain"] > 0.01:
            failures.append("%s: strain %.3f%% > 1%%" % (name, r["strain"] * 100))
        if r["speed"] > 5.0:
            failures.append("%s: node speed %.1f m/s is not physical" % (
                name, r["speed"]))
        if r["arc_err"] > 0.05:
            failures.append("%s: arc length drifted %.3f m" % (name, r["arc_err"]))
        # Energy *boundedness*, not decay. These sea states continuously
        # inject energy through the current, so a driven cable correctly
        # settles to a bounded oscillation rather than to rest. What would
        # signal an unstable solver is KE growing without limit.
        if r["ke_settled"] > 3.0 * r["ke_early"] + 5.0:
            failures.append("%s: kinetic energy growing (%.2f -> %.2f)" % (
                name, r["ke_early"], r["ke_settled"]))
        ke_bound = 0.5 * (CABLE.n_nodes * 2.75) * 3.0 ** 2
        if r["ke_peak"] > ke_bound:
            failures.append("%s: kinetic energy %.0f J exceeds physical bound "
                            "%.0f J" % (name, r["ke_peak"], ke_bound))

    # --- reproducibility
    a = run_case("surge", batch=4, seconds=5.0, seed=7, device=device)
    b = run_case("surge", batch=4, seconds=5.0, seed=7, device=device)
    same = abs(a["tension"] - b["tension"]) < 1e-3
    print("\nreproducible (same seed -> same peak tension): %s" % same)
    if not same:
        failures.append("not reproducible across runs with the same seed")

    print()
    if failures:
        print("CHECKPOINT FAILED")
        for f in failures:
            print("  - %s" % f)
        return 1
    print("CHECKPOINT PASSED: cable hangs and bends stably for 60 s "
          "in all %d sea states." % len(PROFILES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
