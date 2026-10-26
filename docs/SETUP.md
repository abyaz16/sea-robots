# Setup

## What you need

Exactly one credential: a **Nebius API key**. There is no NVIDIA token
required, because Nemotron is served *by* Nebius Token Factory, and that
single call satisfies both hackathon requirements at once — a runtime call to
Token Factory, and use of an NVIDIA open-source model.

| Credential | Required? | Where |
|---|---|---|
| `NEBIUS_API_KEY` | **yes** | [tokenfactory.nebius.com](https://tokenfactory.nebius.com) → API Keys |
| NVIDIA NGC / build.nvidia.com key | no | not used; the call goes through Nebius by design |
| Hugging Face token | optional | only for GR00T / Cosmos weights (gated repos) |
| Tavily key | no | separate bonus prize, unrelated to this project |

## Install

```bash
pip install -r requirements.txt
```

## Configure

```bash
# bash
export NEBIUS_API_KEY=...
export NEBIUS_MODEL=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B   # optional
```

```powershell
# PowerShell
$env:NEBIUS_API_KEY = '...'
```

Defaults, if you set nothing else:

- base URL `https://api.tokenfactory.nebius.com/v1/`
  (use `https://api.tokenfactory.us-central1.nebius.com/v1` on us-central1)
- model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B`

## Verify the live call

```bash
python -m mission.nemotron
```

Prints the request, the raw Nemotron response and the measured latency, and
appends a record to `data/runs/nemotron_calls.jsonl`. **It exits non-zero if
the decision came from the offline planner rather than the model**, so it
cannot pass by accident. That log file is the submission's evidence that a
real runtime call happened.

## Cost

The mission layer runs at 0.25 Hz: roughly 15 calls per episode at ~500
tokens. On Nano-30B ($0.06 / $0.24 per million tokens) the entire project —
including a few hundred evaluation episodes — costs **well under $0.20**.

This matters practically: you do not need the hackathon's $50 in promo
credits. The ~$1 trial credit on a new account is already several times what
the project consumes. What you need is *an account*, not a balance.

### Which model

| Model ID | Context | In / Out per 1M | Notes |
|---|---|---|---|
| `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | 262K | $0.06 / $0.24 | default; ample for a short structured decision |
| `nvidia/nemotron-3-super-120b-a12b` | 256K | $0.30 / $0.90 | native function calling, strictest JSON |
| `nvidia/Nemotron-3-Ultra-550b-a55b` | 1024K | $1.00 / $3.00 | unnecessary here |

Model IDs change; `GET /v1/models` is authoritative.

## Running without a key

Every component runs offline. `mission/nemotron.py` falls back to a
deterministic rule-based planner, and the simulation, training and
evaluation all complete normally.

This is a convenience for development, **not** a substitute for the live
call. Offline decisions are logged with `source: "offline-rule-based"` and
`live: false`, and are never attributed to Nemotron anywhere in the
submission. The rule planner also serves as an honest control: if Nemotron's
decisions do not beat a dozen hand-written rules, the mission layer is
decoration and the write-up should say so.

## If Token Factory asks for a card

Reported by several participants: onboarding can demand a card and autopay
mandate. The contest rules state "NO PURCHASE OR PAYMENT NECESSARY TO ENTER
OR WIN", so this is onboarding friction rather than a contest requirement.
Things that have worked:

1. Ask the organisers — there is a Devpost forum thread on exactly this, and
   the Builders Program runs office hours.
2. A debit card with **international transactions enabled** (the mandate
   usually takes only a small refundable authorisation). Virtual cards from
   Revolut, Wise or Payoneer also work; RuPay often fails on international
   recurring mandates.

## Training

```bash
# local, CPU
python -m policy.train_ppo --steps 2000000 --envs 256 --threads 8

# Nebius AI Cloud GPU node
python -m policy.train_ppo --steps 10000000 --envs 4096 --device cuda
```

Identical code path; only `--envs` and `--device` change. Note that the
promo credits are **Token Factory inference credits and do not cover AI
Cloud GPU compute**, which is billed separately.

## Reproduce the results

```bash
python sim/config.py              # the complete physical model
python -m tests.test_stability    # 60 s stability in 6 sea states
python -m tests.test_lift         # ROV <-> cable gripper coupling
python -m tests.test_spectrum     # is the disturbance fast enough to matter?
python -m eval.run_eval           # 3 control modes x 6 sea states
python -m eval.report             # results table + degradation curve
```
