# Project Leviathan

**Autonomous subsea cable repair, trained and evaluated in simulation.**

Submission to the [Nebius x NVIDIA Global AI Hackathon](https://nebiusglobalaihackathon.devpost.com) — **Physical AI** track.

---

## The claim, in one paragraph

The internet runs on undersea fibre. It breaks around 200 times a year, mostly
from fishing gear and ships' anchors. Today's repair hauls the cable 3 km to a
ship's deck so humans can rejoin it: $1–3M and roughly six weeks per fault,
against a global fleet of only ~20–25 dedicated repair vessels.

Leviathan does the repair **on the seabed**. Three ROVs lift and stabilise the
cable while a fourth mates a wet-mate connector. Nothing is hauled up.

The technical argument is narrow and testable: **a 600 ms round trip to a
surface pilot cannot stabilise a 3.5 kg/m armoured cable against a current
surge, so the control loop has to live on the vehicle.** This repo proves that
with a measurement, not an assertion — identical physics, identical sea
states, one variable: where the control loop runs.

> **Wet-mate connectors, not fusion splicing.** Fibre fusion needs clean, dry
> conditions, which is the entire reason cables get hauled to the surface.
> Wet-mate connectors are mature, deployed subsea technology. Leviathan plugs
> the cable together; it does not weld it.

## Architecture: two loops

The split between the two loops *is* the contribution.

```
┌─ SLOW LOOP — every 4 s — NVIDIA Nemotron via Nebius Token Factory ─┐
│  in:  phase, swarm state summary, tension history, anomalies        │
│  out: {phase, role_assignment, abort?, replan_reason}               │
│  Mission reasoning. Never touches a thruster.                       │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ phase + roles
┌─ FAST LOOP — 50 Hz — onboard learned policy (PPO, trained on Nebius)┐
│  in:  own pose/vel, grip-point error, local tension, peers,         │
│       current estimate, phase one-hot            (~52 dims)         │
│  out: 4-DOF wrench demand ∈ [-1,1]                                  │
│  Station-keeping, tension limiting, formation. No network.          │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ forces
┌─ PHYSICS — 120 Hz — batched XPBD in PyTorch, GPU-parallel ──────────┐
│  cable rod · current field · drag + added mass · thrusters          │
└─────────────────────────────────────────────────────────────────────┘
```

Fast reactions cannot wait for the mission layer. The mission layer does not
need to be fast. Any other division of labour is either too slow to hold the
cable or too myopic to recover from a lost grip.

## Why this is Physical AI, not an app

The fast loop is a **trained policy whose weights encode hydrodynamic
response** — not a prompt, not a PID. Nemotron handles only slow mission
reasoning. The physics is a reduced-order but genuine rigid/flexible-body
simulation: Morison drag with cross-flow/axial split, added mass, a
divergence-free turbulent current field, XPBD inextensibility, and thruster
allocation with real saturation.

## Two failure modes, not one

Most cable models track break tension. Leviathan tracks both limits:

| Failure | Threshold | Why it matters |
|---|---|---|
| Over-tension | 8 kN working / 30 kN break | Strains the fibre, raises attenuation permanently |
| **Over-bending** | 1.0 m minimum bend radius | **Destroys the fibre with no visible damage to the cable** |

Over-bending is the interesting one, because it makes the control problem a
genuine trade-off: lifting harder reduces sag, but a flatter span tightens the
bend at the lift points. There is a narrow safe band, and the policy has to
hold it while being shoved by a current it can only partly observe.

## Status

Day 1 of 7 complete. See [docs/PLAN.md](docs/PLAN.md) for the schedule,
[docs/LATENCY.md](docs/LATENCY.md) for the control argument and the
measurement that tests it, and [docs/LIMITS.md](docs/LIMITS.md) for what this
model does and does not claim.

**Day 1 result.** The cable hangs and bends stably for 60 s in all six sea
states: segment strain under 0.08%, arc length conserved to 7 mm over 30 m,
bounded energy, bit-reproducible from a seed. Two ROVs lift it off the seabed
within both limits, with the gripper coupling conserving momentum exactly.
And the disturbance spectrum has been measured against the latency corner
frequencies, because that measurement could have falsified the whole project:
**37-39% of disturbance energy lies above what a 600 ms loop can correct.**

| Day | Deliverable | State |
|---|---|---|
| 1 | Cable rod, current field, ROV thrusters, stability checkpoint | **passed** |
| 2 | Gym env, phases, reward, first PPO run | |
| 3 | Full curriculum through surge gusts | |
| 4 | Nemotron mission layer, mate phase | |
| 5 | Teleop baselines + full eval table | |
| 6 | Cosmos sonar synthesis, R3F replay viewer | |
| 7 | Record, write up, submit | |

## Repo layout

```
sim/        config.py  cable.py  water.py  rov.py  env.py  sonar.py
policy/     train_ppo.py  nets.py  curriculum.py
mission/    nemotron.py        — Nebius Token Factory client
baselines/  teleop.py          — delayed PD human proxy
eval/       run_eval.py  report.py
viz/        React Three Fiber replay viewer
tests/      test_stability.py  — Day-1 hard checkpoint: 60 s, 6 sea states
            test_lift.py       — ROV<->cable gripper coupling
            test_spectrum.py   — is the disturbance fast enough to matter?
docs/       PLAN.md  LATENCY.md  LIMITS.md  SUBMISSION.md  FEEDBACK.md
```

Every physical constant lives in [`sim/config.py`](sim/config.py) so a
reviewer can audit every assumption in one file. `python sim/config.py`
prints the whole model spec.

## Quickstart

```bash
pip install -r requirements.txt
python sim/config.py              # print the physical model
python -m tests.test_stability    # Day-1 checkpoint: 60 s in 6 sea states
python -m tests.test_lift         # gripper coupling: two ROVs lift the cable
python -m tests.test_spectrum     # does loop latency actually matter here?
```

## License

Apache-2.0. See [LICENSE](LICENSE).
