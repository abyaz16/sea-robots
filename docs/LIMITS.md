# What this model does and does not claim

Stating boundaries is not a weakness in a simulation result; it is the only
thing that makes the result readable. Everything below is a deliberate
modelling choice, not an oversight. Where a choice could plausibly change the
headline conclusion, that is said explicitly.

## The headline claim, stated precisely

> In this simulation, a control policy running at 50 Hz on the vehicle
> completes seabed cable mating at a higher success rate, with lower peak
> cable tension and fewer bend-radius violations, than a delayed-feedback
> controller representing a surface pilot — across identical physics,
> identical sea states, and including sea states never seen in training.

What that claim is **not**: it is not a demonstration on hardware, not a
validated hydrodynamic prediction of absolute forces, and not evidence that
any particular real repair would succeed.

## Reduced-order physics

| Modelled | Not modelled | Does it threaten the conclusion? |
|---|---|---|
| Morison drag, cross-flow/axial split | Vortex-induced vibration, lift forces | No — VIV would load *both* control modes |
| Constant diagonal added mass | Frequency-dependent added-mass tensor | No — affects both modes equally |
| Divergence-free synthetic turbulence from 6 Fourier modes | Full Navier–Stokes / LES | **Partly** — see "the one that matters" below |
| XPBD inextensibility + one-sided bending | Torsion, axial stiffness hysteresis, armour-wire mechanics | No — torsion is not in the control loop |
| Seabed non-penetration + Coulomb friction | Burial suction, soil strain-rate effects, trenching | **Partly** — de-burial load could dominate real tension |
| 4-DOF ROV (surge, sway, heave, yaw) | Roll and pitch; assumed trim-stabilised | No — work-class ROVs are strongly trim-stable |
| 6-thruster allocation with real saturation and lag | Thruster–thruster wake interaction, cavitation | No |
| — | Tether drag from a surface umbilical | **Partly** — would *worsen* the teleop baseline, not help it |

### The one that matters

The turbulence spectrum is synthetic. Its energy is concentrated in
wavelengths of 2–40 m and periods of 3–25 s, chosen because that is the band
that actually accelerates a 1.5 t vehicle. If real benthic turbulence carried
substantially more energy at periods *below* 1.2 s, the advantage of a 50 Hz
loop over a 2 Hz loop would be **larger** than reported, not smaller. If it
carried much less energy in the 3–25 s band, the advantage would shrink.
The direction of the result is therefore robust; the magnitude is not.

## The teleop baseline is a proxy, and here is exactly what it is

The comparison is against a **delayed PD controller**, not against a human.
It is specified as:

- observations delayed by 600 ms (or 150 ms in the shallow-water case)
- commands updated at 2 Hz, held between updates
- an additional 250 ms reaction lag before a disturbance affects the command
- gains tuned by search on the *same* training sea states the policy saw, so
  the baseline is not handicapped by poor tuning

A skilled human pilot has feedforward skill a PD controller does not: they
anticipate a recurring swell. A human also fatigues, loses situational
awareness in low visibility, and cannot attend to four vehicles at once —
none of which the proxy models. **The proxy is neither strictly a lower nor
strictly an upper bound on human performance.** It isolates one variable:
loop latency. That is the variable the project is about.

## Wet-mate, not fusion

Leviathan mates a wet-mate connector. It does not fusion-splice fibre, which
requires clean dry conditions and is the actual reason cables are hauled up.
Wet-mate connectors are deployed subsea technology, but they carry real
optical insertion loss and a finite mating cycle count, and a repair built on
them has a different loss budget than a fusion splice. That engineering
trade-off is outside the scope of this project.

## The connector mate itself is simplified

Success is defined geometrically: the connector is brought within a position
and alignment tolerance of the receptacle and held there for a dwell period.
Real wet-mate insertion involves guide-cone contact mechanics, seal friction,
and latch engagement. Modelling that would change the *mate* phase; it would
not change the *stabilisation* problem, which is what the comparison measures.

## No hardware

Nothing here has been wet-tested. The autonomy stack has never seen a real
sonar return, real soil, or a real connector. The contribution is a
simulation result and an architecture argument, and it should be read as
exactly that.

## Resident robots are an assumption about deployment, not a result

The pitch assumes the vehicles are already on site, docked in a seabed
garage. Resident subsea vehicles in docking stations are real and deployed in
offshore energy. Nothing in this simulation tests the docking, charging, or
transit-to-fault phases.

## Reproducibility

Every result is seeded. `sim/config.py` holds every physical constant in one
module, and `python sim/config.py` prints the complete model spec. The
evaluation script re-derives every number in the results table from stored
trajectories, so any figure in the submission can be traced to a run.
