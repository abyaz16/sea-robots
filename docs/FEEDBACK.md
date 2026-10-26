# Feedback on Nebius and NVIDIA tooling

Kept as a running log from the first hour of the build rather than written
from memory at the end, because the useful feedback is the friction you
forget once you have worked around it. Each entry notes what we expected,
what happened, and what would have saved the time.

Format: `[date] [component] — observation → suggestion`

---

## Nebius AI Cloud (policy training)

_(entries added as we train — Day 1 is local physics development)_

- **[Oct 23] Account / GPU access** — Pending. Will record time-to-first-GPU
  here, since for a 7-day build the latency from signup to a usable device is
  the single most consequential number about a cloud.

## Nebius Token Factory (Nemotron inference)

- **[Oct 23] Endpoint shape** — Expecting an OpenAI-compatible endpoint, so
  the mission client is written against the `openai` SDK with a custom
  `base_url`. If that assumption holds it is the right call and worth saying
  loudly in the docs; being able to point an existing SDK at it removes an
  entire integration step.

## NVIDIA Nemotron

- **[Oct 23] Structured output** — The mission layer needs strict JSON
  (`{phase, role_assignment, abort, replan_reason}`) because its output drives
  a control system, not a chat window. Noting here whether constrained /
  schema-guided decoding is available on the hosted endpoint, since the
  alternative — parse, validate, retry — is the difference between a mission
  layer you can trust in a 50 Hz loop's supervisor and one you cannot.

## Physics / simulation friction (not a Nebius or NVIDIA issue, recorded for completeness)

- **[Oct 23] PyTorch as a physics engine** — Batched XPBD in PyTorch works
  and is the reason thousands of parallel environments are feasible, but the
  constraint solve is dominated by many small `index_add_` scatters rather
  than by arithmetic. Switching from out-of-place to in-place scatter was a
  measurable win. A fused scatter-with-colouring primitive would help anyone
  doing position-based dynamics on GPU; this is a real gap between "PyTorch
  for networks" and "PyTorch for physics".

---

## Summary (to be completed before submission)

The three things that most helped:

1. _tbd_
2. _tbd_
3. _tbd_

The three things that cost the most time:

1. _tbd_
2. _tbd_
3. _tbd_
