# Demo video — shot list and script

**Hard rules:** under 3 minutes, public on YouTube, no copyrighted music, and
for the Physical AI track **at least 60 seconds showing the system
operating**. The plan below gives 70 s of operating footage (0:28–1:38).

Lead with the finding. Judges watch dozens of these; the first ten seconds
decide whether they watch the rest.

| Time | Shot | Narration (read, don't improvise) |
|---|---|---|
| 0:00–0:12 | The stat, then the results table on screen | "The internet runs on undersea fibre. It breaks about two hundred times a year. Each repair costs one to three million dollars and takes six weeks — because the cable is hauled three kilometres to a ship's deck so humans can rejoin it." |
| 0:12–0:28 | Diagram: today's method, cable going up to the ship | "Leviathan does the repair on the seabed. Three robots lift and stabilise the cable, a fourth mates a wet-mate connector. Nothing is hauled up." |
| **0:28–1:00** | **Replay, teleop @ 600 ms, surge condition.** Let it run. Tension bar climbs amber then red; connector gap never closes; FAILED | "This is a pilot on the ship, with a six-hundred-millisecond round trip. A surge gust hits. By the time the correction arrives, the cable is already over-tensioned. The run fails." |
| **1:00–1:38** | **Replay, onboard autonomy, same surge, same seed.** Formation holds; tension stays green; gap closes; MATED | "Same gust. Same cable. The only change is where the control loop runs. On the vehicle, at fifty hertz, the policy holds the span inside both limits and the connector mates." |
| 1:38–1:52 | Architecture diagram, two loops | "Two loops. Nemotron on Nebius Token Factory does mission reasoning every four seconds — phases, roles, when to abort. It never touches a thruster. A trained policy runs the thrusters at fifty hertz, onboard, with no network." |
| 1:52–2:12 | Training curves + "trained on Nebius"; then the held-out results | "The policy was trained with PPO on batched GPU physics. These sea states were never in training — the result holds on held-out currents." |
| 2:12–2:32 | Degradation curve | "Success against current speed. Teleop falls off a cliff. Autonomy degrades gracefully. That gap is the whole argument, and it is measured, not asserted." |
| 2:32–2:50 | Limits slide, read plainly | "Honest limits: reduced-order hydrodynamics, wet-mate connectors not fusion splicing, the teleop baseline is a delayed PD controller and not a human, and no hardware has been wet-tested." |

**The 0:28–1:38 pair is the whole video.** One run fails, one succeeds, the
viewer sees exactly why, and the only difference is loop latency.

## Production notes

- Record the viewer at **1920x1080**, browser zoomed so the HUD is legible.
  The side-by-side layout is built for exactly this shot.
- Press **Camera** once to get the low wide angle for the failure run — the
  sag and the over-bend read better from near the seabed.
- Use **0.5x** speed for the gust moment in both runs so the viewer can see
  the tension bar move.
- Keep the mouse still. Nothing cheapens a demo faster than a jittery
  orbit camera.
- Narrate over silence or a CC0 ambient bed. **No copyrighted music** — this
  is an explicit disqualification risk, not a style note.
- Export the two figures at 160 dpi (the report script already does) and cut
  to them full-frame rather than screenshotting the markdown.

## Checklist before upload

- [ ] Under 3:00 total
- [ ] ≥60 s of the system operating (0:28–1:38 gives 70 s)
- [ ] "Trained on Nebius AI Cloud" visible on screen
- [ ] Nemotron named on screen, with a real decision trace visible
- [ ] Results table and degradation curve both shown
- [ ] Limits stated out loud, not just on a slide
- [ ] Visibility set to **Public** (not Unlisted — the rules say public)
- [ ] No copyrighted audio
