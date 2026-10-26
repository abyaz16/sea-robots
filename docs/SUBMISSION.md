# Project Leviathan — submission

**Track:** Physical AI
**Hackathon:** Nebius x NVIDIA Global AI Hackathon
**Licence:** Apache-2.0

---

## 1. The problem, specifically

The internet runs on roughly 1.5 million km of submarine fibre. It breaks
about **200 times a year**, most often from fishing gear and ships' anchors.

Each repair:

- costs **$1–3M**
- takes around **six weeks**, most of it waiting for a vessel
- depends on a global fleet with only **~20–25 vessels dedicated to repair**,
  much of it ageing

The reason it is slow and expensive is not the join itself. It is that
**the cable is hauled three kilometres to the surface** so people can work on
it in a clean dry room, then lowered back. The ship, the winches and the
waiting are the cost.

## 2. What Leviathan does

Repairs the cable **where it lies**. Three ROVs lift and stabilise the span
against deep-sea currents while a fourth mates a **wet-mate connector**.
Nothing surfaces.

> **Wet-mate connectors, not fusion splicing.** Fibre fusion needs clean dry
> conditions — which is precisely why cables get hauled up. Wet-mate
> connectors are mature, deployed subsea technology. Leviathan plugs the
> cable together; it does not weld it. Saying this correctly matters, because
> a submission claiming underwater fusion splicing would be claiming
> something that does not work.

Deployment assumption: the vehicles are **resident** on the seabed in a
docking station, as already done in offshore energy. Nothing here tests
docking, charging or transit.

## 3. The technical claim

**A 600 ms round trip to a surface pilot cannot stabilise a 3.5 kg/m armoured
cable against a current surge. The control loop has to live on the vehicle.**

This is a control-theoretic claim with a derivation and a measurement, not a
slogan. A transport delay `L` caps closed-loop bandwidth near
`f = 1/(2πL)`, and the effective delay includes the command-hold half-period
and operator reaction:

| Mode | Effective delay | Bandwidth |
|---|---|---|
| A — teleop @ 600 ms | 1100 ms | **0.14 Hz** |
| B — teleop @ 150 ms | 650 ms | **0.24 Hz** |
| C — onboard autonomy | 20 ms | **7.96 Hz** |

Mode C has ~55x the usable bandwidth of mode A.

A bandwidth advantage is only worth anything if the disturbance has energy up
there, so **we measured the thing that could have falsified the project**
(`tests/test_spectrum.py`): **23.6–38.8% of disturbance force energy lies
above mode A's corner frequency.** Full derivation, the measurement, and the
two bugs that first made it look refuted, are in
[LATENCY.md](LATENCY.md).

## 4. Architecture — two loops

```
┌─ SLOW — every 4 s — NVIDIA Nemotron via Nebius Token Factory ───────┐
│  in:  phase, swarm summary, tension trend, grip integrity, anomalies │
│  out: {phase, roles, abort, reason}  — strict JSON, validated        │
│  Mission reasoning. Never touches a thruster.                        │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ phase + roles
┌─ FAST — 50 Hz — onboard learned policy (PPO, trained on Nebius) ─────┐
│  in:  63-dim egocentric obs — own pose/vel, grip-point error, local  │
│       tension, bend margin, peers, flow estimate, phase, role        │
│  out: 4-DOF wrench demand ∈ [-1,1], allocated to 6 thrusters         │
│  Station-keeping, tension limiting, formation. No network.           │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ forces
┌─ PHYSICS — 150 Hz — batched XPBD in PyTorch, GPU-parallel ───────────┐
│  cable rod · turbulent current + VIV · Morison drag + added mass     │
└──────────────────────────────────────────────────────────────────────┘
```

The division is the contribution. A mission layer at 0.25 Hz is fine. A
*control* layer at 0.25 Hz destroys the cable. And the fast policy cannot do
the mission layer's job: with γ=0.995 at 50 Hz its effective horizon is about
four seconds by construction.

## 5. Two failure modes, not one

| Failure | Threshold | Why it matters |
|---|---|---|
| Over-tension | 8 kN working (30 kN break) | Strains the fibre, raises attenuation permanently |
| **Over-bending** | 1.0 m minimum bend radius | **Destroys the fibre with no visible damage to the cable** |

Modelling over-bending is what makes this a real control problem rather than
a lifting exercise: **lifting harder flattens the sag, which tightens the
bend at the lift points.** There is a narrow safe band, and the policy has to
hold it while being shoved by a current it observes only through noisy,
intermittent sonar.

Measured in the oracle run: peak tension 30% of working limit with min bend
radius 2.11 m — i.e. the bend limit is the binding constraint, only 2.1x
away, while tension sits comfortable. That is the opposite of what a
tension-only model would tell you, and it is the kind of thing a simulation
is for.

## 6. The experiment

Three control modes, **identical physics, identical paired seeds**, one
variable: where the control loop runs. Six sea states, three of them **never
seen in training**. Plus a zero-latency full-state oracle as an upper bound,
to separate "the loop is slow" from "the task is hard".

Results, tables, paired significance tests and the degradation curve:
**[RESULTS.md](RESULTS.md)** (generated by `python -m eval.report`).

## 7. Hackathon requirements

| Requirement | How it is met |
|---|---|
| Runtime call to Nebius Token Factory | Nemotron mission layer, every 4 s. Every call logged to `data/runs/nemotron_calls.jsonl` with latency and token usage. `python -m mission.nemotron` verifies it and **exits non-zero if the response came from the offline fallback** |
| ≥1 NVIDIA open-source model | **NVIDIA Nemotron** (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`) |
| Runs on Nebius AI Cloud | PPO training: `python -m policy.train_ppo --envs 4096 --device cuda` |
| Physical AI track | Learned control policy acting in a simulated body, with policy evaluation |
| Public repo, OSI licence | Apache-2.0 |
| README + setup | [README](../README.md), [SETUP.md](SETUP.md) |
| Demo video <3 min, ≥60 s operating | [VIDEO.md](VIDEO.md) |
| Tool feedback | [FEEDBACK.md](FEEDBACK.md) |

## 8. Honest limits

Full version: **[LIMITS.md](LIMITS.md)**. The short form:

- Reduced-order hydrodynamics. No CFD, no VIV lock-in amplification, no
  vortex-induced vibration of the vehicles, no umbilical tether load.
- The turbulence spectrum is synthetic, and it is **the most load-bearing
  assumption in the project** — the above-corner disturbance energy comes
  mostly from it, not from VIV as we first assumed.
- **The teleop baseline is a delayed PD controller, not a human.** It is
  neither a strict upper nor a strict lower bound on a pilot. It isolates
  loop latency, which is the variable under test. Its gains are tuned per
  latency setting, so it is not handicapped.
- Wet-mate connector mating is modelled **geometrically** (position,
  alignment, dwell), not as contact mechanics.
- Approach and gripper closure are initialised, not learned. The evaluated
  task is LIFT → STABILISE → MATE.
- Soil is non-penetration plus Coulomb friction. Real burial suction can
  dominate the de-burial load.
- **No hardware.** Nothing has been wet-tested. This is a simulation result
  and an architecture argument, and should be read as exactly that.

## 9. Reproducing everything

```bash
pip install -r requirements.txt
python sim/config.py              # every physical constant, in one place
python -m tests.test_stability    # 60 s stability, 6 sea states
python -m tests.test_lift         # ROV <-> cable gripper coupling
python -m tests.test_spectrum     # the falsification test
python -m mission.nemotron        # verify the live Nebius call
python -m policy.train_ppo        # train the fast loop
python -m eval.run_eval --runs 50 # the experiment
python -m eval.report             # table + figures
```

Every number in the write-up traces to a seeded run.
