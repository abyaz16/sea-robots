# Why loop latency is the whole argument

This document exists because the project's central claim is a *control*
claim, and control claims are easy to assert and easy to get wrong. It
records the derivation, the measurement that tests it, and the mistake we
made first.

## The claim

A surface pilot cannot stabilise a lifted armoured cable against seabed
disturbance, because the round-trip latency of the control loop caps the
bandwidth below the disturbance bandwidth. An onboard policy at 50 Hz does
not have that cap.

## The derivation

A pure transport delay `L` in a feedback loop contributes phase lag `ωL`. For
a stable loop with usable phase margin you need roughly `ωL < 1 rad`, so
achievable closed-loop bandwidth is

```
ω_max ≈ 1 / L        f_max ≈ 1 / (2π L)
```

Above `f_max` feedback stops attenuating the disturbance and starts
amplifying it. The effective delay is not just the network round trip; it is

```
L_eff  =  observation delay  +  command-hold half-period  +  reaction lag
```

The command-hold term matters and is easy to forget: a controller issuing
commands at `f_cmd` holds each one for `1/f_cmd`, which is an average extra
delay of `1/(2 f_cmd)`.

| Mode | Obs delay | Command rate | ZOH half-period | Reaction | `L_eff` | `f_max` |
|---|---|---|---|---|---|---|
| A — teleop @ 600 ms | 600 ms | 2 Hz | 250 ms | 250 ms | **1100 ms** | **0.14 Hz** |
| B — teleop @ 150 ms | 150 ms | 2 Hz | 250 ms | 250 ms | **650 ms** | **0.24 Hz** |
| C — onboard autonomy | 20 ms | 50 Hz | 10 ms | — | **20 ms** | **7.96 Hz** |

Mode C has roughly **55x** the usable bandwidth of mode A. That is the
entire technical argument, and it is arithmetic, not opinion.

## The test

A bandwidth advantage only matters if the disturbance has energy up there.
`tests/test_spectrum.py` measures the power spectrum of the hydrodynamic
force on a lifted cable span and reports the fraction of fluctuating energy
above each mode's corner frequency. If that fraction were negligible, the
latency claim would be unsupportable and we would have to say so.

Measured, with VIV disabled, in the `moderate` sea state:

```
sea       VIV     >0.14Hz (A)   >0.24Hz (B)  rms force N
moderate  off           36.9%         18.7%          3.6
moderate  on            38.8%         19.5%          3.7
strong    off           23.6%         15.5%          3.9
strong    on            26.1%         18.3%          4.1
surge     off           36.6%          5.9%          5.3
surge     on            37.0%          7.7%          5.5
```

Read the shares together with the rms, because a share is not a load. The
absolute above-corner force, `sqrt(fraction) * rms`, is what the controller
actually has to fight:

| sea | above-corner force, mode A | above-corner force, mode B |
|---|---|---|
| moderate | 2.3 N | 1.6 N |
| strong | 2.1 N | 1.8 N |
| **surge** | **3.3 N** | **1.5 N** |

`strong` has a *lower* above-corner share than `moderate` (23.6% vs 36.9%)
purely because its faster mean flow adds low-frequency energy that dilutes
the fraction; its absolute fast load is similar. The share alone would have
been misleading.

### A prediction, made before any policy is trained

Look at the `surge` row: **36.6% of its energy is above mode A's corner but
only 5.9% is above mode B's.** The surge gust energy sits almost entirely in
the narrow band *between the two teleop corners*, 0.14–0.24 Hz.

The physics therefore predicts the shape of the results table in advance:
on the surge condition, **mode A should fail where mode B largely copes**,
and the A → B gap should be wider there than in any other sea state. If the
trained comparison does not show that ordering, something is wrong with the
experiment rather than with the prediction — which makes this a check on our
own eval, not just a forecast.

**37–39% of the disturbance energy sits above what a 600 ms loop can
correct.** The claim survives its own test.

Note what this says about VIV: it contributes about **2 percentage points**,
not the bulk. We added VIV expecting it to be the dominant fast load and it
is not, because finite spanwise coherence (roughly 15 independent shedding
cells over a 30 m span) cancels most of the net force. The above-corner
energy comes mainly from the **turbulent current band itself**, which spans
0.04–0.33 Hz and therefore straddles the 0.14 Hz corner.

That makes the turbulence spectrum the single most load-bearing modelling
assumption in the project, which is why it is listed first under "what would
change the conclusion" and first in `LIMITS.md`. It would have been easy to
report the VIV number as the headline; it is not what the measurement says.

## The mistake we made first, recorded on purpose

Our first version of this test reported only 0.4–1.6% of energy above the
corner, and briefly looked like it had refuted the project. The error was in
the test, not the physics: we used `f = 1/(2L)` instead of `f = 1/(2πL)`, and
omitted the command-hold and reaction terms from the effective delay. That
put the corner at 0.83 Hz instead of 0.14 Hz — about 6x too strict — and
almost all of the disturbance energy fell below it by construction.

A second bug compounded it. Vortex-shedding lift acts perpendicular to both
the cylinder axis and the flow, so for a horizontal cable in a horizontal
current the VIV force is **vertical**. The probe sampled only horizontal
components and therefore saw almost none of the fast forcing it was built to
detect.

Both are recorded here because "we measured the thing that would have
falsified our claim, and here is the version of the measurement that was
wrong" is a stronger statement than a clean number with no history.

## Vortex-induced vibration

VIV is the fast disturbance in this problem and is modelled explicitly
(`sim/cable.py::_viv_force`). Shedding frequency follows Strouhal scaling,
`f = St·u/D`; at `St = 0.2`, `D = 0.05 m` and `u = 0.3 m/s` that is **1.2 Hz**
— two decades above the turbulent current band, and far above any teleop
corner frequency.

Modelling choices:

- `Cl = 0.7`, a mid-range cross-flow lift coefficient near lock-in
- phase is **integrated**, not evaluated as `sin(2π f(t) t)`, because the
  local shedding frequency varies with the flow and the naive form produces a
  spurious chirp
- shedding is given a finite spanwise coherence length (2 m, ~40 diameters).
  Fully coherent shedding would overstate the net load on a 30 m span by
  roughly `sqrt(N_cells)`; fully random would understate it
- lock-in amplification, where shedding locks to a structural mode and
  amplitude grows several-fold, is **not** modelled. That omission is
  conservative: it makes teleop look better than it would be.

## What would change the conclusion

Honest list, in order of how much it would hurt:

1. If real benthic turbulence carried much less energy in the 0.1–1 Hz band
   than this model, the margin would shrink. The direction of the result
   would survive; the magnitude would not.
2. If a human pilot's feedforward anticipation of a periodic swell were
   strong, mode A would beat its own bandwidth limit on the periodic
   component. It would not help against turbulence or VIV, which are not
   predictable.
3. If the task tolerance were loose enough that 0.14 Hz bandwidth sufficed,
   latency would stop mattering. This is why the mate tolerance is specified
   and reported, not tuned to taste.
