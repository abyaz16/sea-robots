# Seven days

Submission deadline: **Fri 30 Oct 2026, 17:00 UTC** (10:00 PDT). Target
submit date is **29 Oct**, one day spare. Judging runs 1–15 Dec, so anything
hosted must stay up until 15 Dec.

| Day | Date | Work | Hard checkpoint |
|---|---|---|---|
| 1 | Oct 23 | Cable rod, current field, ROV thrusters, Nebius GPU access | Cable hangs and bends stably for 60 s in all 6 sea states |
| 2 | Oct 24 | Gym env, phases, reward. Curriculum stage 1. First PPO run | Reward rising; two ROVs lift the cable without breaking it |
| 3 | Oct 25 | Full curriculum through surge gusts. Long training run | Policy holds station in turbulence |
| 4 | Oct 26 | Nemotron mission layer. Mate phase | Full episode completes end to end |
| 5 | Oct 27 | Teleop baselines + full eval. Lock every number | **The comparison table exists** |
| 6 | Oct 28 | Cosmos sonar, R3F replay polish, GR00T if time | Replay renders beautifully |
| 7 | Oct 29 | Record, edit, write up, submit | Submitted |

**Day 5 is sacred.** If Oct 27 arrives without the comparison table, stop
building features and produce it. A plain project with the table beats a
beautiful one without it.

## Scoring, and what follows from it

Judging is four **equally weighted** criteria. That 25% weight on Design
changes the priority order from the obvious one:

| Criterion | 25% each | What earns it here |
|---|---|---|
| Technological Implementation | How well built, and how well it uses Nebius + an NVIDIA open model | Batched XPBD physics, PPO on Nebius AI Cloud, Nemotron via Token Factory |
| **Design** | Completeness and coherence of the product experience | **The R3F replay viewer is not polish — it is a quarter of the score** |
| Potential Impact | Credibility and specificity of the problem | $1–3M and ~6 weeks per fault, ~200 faults/yr, 20–25 repair vessels |
| Quality of the Idea | Creativity, non-obviousness, grasp of the problem space | Two-loop split; the bend-radius failure mode nobody models |

Consequence: the viewer joins the eval table on the never-cut list.

## Hard requirements (from the official rules)

- [ ] Runtime call to Nebius Token Factory, **or** runs on Nebius AI Cloud — we do both
- [ ] At least one NVIDIA open-source model — Nemotron satisfies this alone
- [ ] Public repo, OSI licence detectable by tooling — Apache-2.0, in place
- [ ] README with setup instructions, covering Nemotron use and how Token Factory helped
- [ ] Demo video **under 3 minutes**, public on YouTube, no copyrighted music
- [ ] ≥1 minute of the system actually operating (Physical AI rule)
- [ ] Written feedback on Nebius and NVIDIA tooling — see `FEEDBACK.md`
- [ ] All materials in English
- [ ] Demo URL not required for Physical AI; if given, must live until 15 Dec

## Cut order, if behind

GR00T → Cosmos sonar synthesis → live WebSocket mode → third current profile
→ fourth ROV.

**Never cut:** the teleop baseline, the eval table, the replay viewer, the
video.

## Risk register

| Risk | Mitigation | State |
|---|---|---|
| Cable sim explodes (stiff constraints) | XPBD compliance, 8 substeps, velocity clamp, arc-length-correct init | Day 1 checkpoint measures it |
| PPO will not converge | Curriculum is mandatory, starting at zero current. If stuck by Day 3, drop to 2 ROVs | |
| CPU-only local box | Physics dev local, all training on Nebius GPU. Throughput measured in the checkpoint | Local run is ~0.7x realtime/env |
| Nemotron adds nothing visible | Script a surge that forces a genuine re-plan, so the mission layer visibly earns its place | |
| GR00T integration eats a day | Already optional; Nemotron alone satisfies eligibility | Drop without hesitation |
