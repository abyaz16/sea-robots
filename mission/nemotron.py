"""The slow loop: mission reasoning with NVIDIA Nemotron on Nebius Token Factory.

What this layer is for
----------------------
It runs every ~4 s and decides *what the swarm should be doing*: which phase
the operation is in, which vehicle holds which grip point, whether to abort
and re-seat a lost grip, and whether conditions are too severe to continue.
It never touches a thruster. Thruster commands belong to the 50 Hz onboard
policy, which cannot wait for a language model and does not need to.

That division is the architectural claim of the project. A mission layer at
0.25 Hz is fine; a control layer at 0.25 Hz destroys the cable.

Why a language model at all
---------------------------
Because the mission decisions are exactly the kind that are awkward to write
as rules: "lifter 2 has lost its grip twice in the last minute, tension is
trending up, and a gust is due -- do we re-seat, redistribute to the
remaining two, or set down and wait?" The space of sensible responses is
large, the inputs are heterogeneous, and the output is a short structured
decision. That is a good fit, and it is genuinely *not* something the PPO
policy can do: the policy has a 4-second effective horizon by construction.

Honesty about the offline path
------------------------------
If no API key is present this module falls back to a deterministic
rule-based planner so the simulation still runs end to end. Every logged
call records `source`, which is either the model ID or
`"offline-rule-based"`. Offline output is never attributed to Nemotron, and
`verify_live_call()` exists so the submission can prove a real runtime call
was made.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Defaults match the Nebius Token Factory catalogue. Nano-30B is the cheapest
# Nemotron and is ample for a short structured decision; Super-120B has
# native function calling if stricter JSON is wanted.
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
OFFLINE = "offline-rule-based"

PHASE_NAMES = ("LIFT", "STABILISE", "MATE", "DONE")

SYSTEM_PROMPT = """\
You are the mission supervisor for an autonomous subsea cable repair swarm \
operating on the seabed at 3 km depth. Four ROVs are involved: three lifters \
holding an armoured fibre-optic cable, and one connector vehicle that mates a \
wet-mate connector at the mid-span node.

You run once every few seconds. You do NOT fly the vehicles: a learned \
onboard policy at 50 Hz does the station-keeping and tension limiting. Your \
job is the slow decision: which phase the operation is in, role assignment, \
and whether to hold, re-seat a grip, or abort.

Hard constraints on the cable:
- working tension 8000 N; exceeding it strains the fibre permanently
- minimum bend radius 1.0 m; violating it destroys the fibre with no visible \
damage to the cable
Lifting harder flattens the sag but tightens the bend at the lift points, so \
there is a narrow safe band. Prefer holding over forcing progress.

Respond with ONLY a JSON object, no prose, no code fences:
{"phase": "LIFT"|"STABILISE"|"MATE"|"HOLD"|"ABORT",
 "roles": {"0": "lifter"|"idle"|"reseat", "1": ..., "2": ..., "3": \
"connector"|"standoff"},
 "abort": true|false,
 "reason": "<=140 chars, state the governing measurement>"}
"""


@dataclass
class SwarmSummary:
    """The compact state handed to the mission layer.

    Deliberately small and already reduced to engineering quantities. Sending
    raw 63-dim observations would be both expensive and useless: the decision
    turns on tension trend and grip integrity, not on instantaneous pose.
    """
    t: float
    phase: str
    cable_height: float
    peak_tension_n: float
    tension_frac_working: float
    tension_trend: float                  # N/s over the last window
    min_bend_radius_m: float
    bend_margin: float                    # R_min / R_limit
    station_err_max_m: float
    grips_held: int
    grip_losses: int
    mate_dist_m: float
    mate_dwell_s: float
    current_speed_est: float
    gust_active: bool
    anomalies: List[str] = field(default_factory=list)

    def as_prompt(self) -> str:
        d = asdict(self)
        lines = ["Mission state:"]
        for k, v in d.items():
            if isinstance(v, float):
                lines.append("  %s: %.3f" % (k, v))
            else:
                lines.append("  %s: %s" % (k, v))
        lines.append("Decide the phase and roles.")
        return "\n".join(lines)


@dataclass
class MissionDecision:
    phase: str
    roles: Dict[str, str]
    abort: bool
    reason: str
    source: str
    latency_ms: float
    raw: str = ""

    @property
    def is_live(self) -> bool:
        return self.source != OFFLINE


def _coerce(obj: Any) -> Optional[MissionDecision]:
    """Validate a model response into a decision, or return None.

    A mission layer whose output drives a control system cannot accept
    whatever the model felt like emitting. Anything that does not validate is
    rejected and retried, and a persistent failure falls back to the rule
    planner rather than passing a malformed phase downstream.
    """
    if not isinstance(obj, dict):
        return None
    phase = str(obj.get("phase", "")).upper()
    if phase not in (*PHASE_NAMES, "HOLD", "ABORT"):
        return None
    roles = obj.get("roles", {})
    if not isinstance(roles, dict):
        roles = {}
    roles = {str(k): str(v) for k, v in roles.items()}
    return MissionDecision(phase=phase, roles=roles,
                           abort=bool(obj.get("abort", False)),
                           reason=str(obj.get("reason", ""))[:200],
                           source="", latency_ms=0.0)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of a response, fences and all."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        t = t[4:] if t.lower().startswith("json") else t
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return None


def rule_based(s: SwarmSummary) -> MissionDecision:
    """Deterministic planner. The offline fallback, and the sanity baseline.

    This is also useful as a control: if Nemotron's decisions do not beat
    these rules, the mission layer is decoration and we should say so.
    """
    t0 = time.perf_counter()
    roles = {"0": "lifter", "1": "lifter", "2": "lifter", "3": "standoff"}
    abort = False

    # Order matters: the conditions that destroy the cable are tested before
    # the ones that merely pause the operation. Checking "near the limit"
    # first would downgrade an already-exceeded limit to a hold.
    if s.tension_frac_working > 1.25 or s.bend_margin < 1.0:
        phase, reason, abort = "ABORT", (
            "cable limit exceeded (tension %.0f%%, bend margin %.2f); set down"
            % (100 * s.tension_frac_working, s.bend_margin)), True
    elif s.grips_held < 3:
        phase, reason = "HOLD", "grip lost (%d/3 held); re-seat before loading" % s.grips_held
        for k in ("0", "1", "2"):
            roles[k] = "reseat"
    elif s.bend_margin < 1.15:
        phase, reason = "HOLD", "bend margin %.2f near limit; ease the lift" % s.bend_margin
    elif s.tension_frac_working > 0.95:
        phase, reason = "HOLD", "tension %.0f%% of working limit" % (100 * s.tension_frac_working)
    elif s.gust_active and s.phase == "MATE" and s.mate_dist_m > 0.3:
        phase, reason = "HOLD", "gust active; defer mate until the surge passes"
    elif s.cable_height < 3.6:
        phase, reason = "LIFT", "height %.2f m below working height" % s.cable_height
    elif s.station_err_max_m > 0.35:
        phase, reason = "STABILISE", "station error %.2f m above tolerance" % s.station_err_max_m
    else:
        phase, reason = "MATE", "span stable; proceed to mate"
        roles["3"] = "connector"

    return MissionDecision(phase=phase, roles=roles, abort=abort, reason=reason,
                           source=OFFLINE,
                           latency_ms=(time.perf_counter() - t0) * 1e3)


class NemotronMission:
    """Nebius Token Factory client for the mission layer.

    Credential-agnostic by design: it reads `NEBIUS_API_KEY`, and if there is
    none it runs the rule planner and says so. That keeps the simulator
    runnable end to end without a key, while making the live path a single
    environment variable away.
    """

    def __init__(self, model: Optional[str] = None,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 log_path: str | Path = "data/runs/nemotron_calls.jsonl",
                 timeout: float = 20.0,
                 max_retries: int = 2,
                 force_offline: bool = False) -> None:
        self.model = model or os.environ.get("NEBIUS_MODEL", DEFAULT_MODEL)
        self.base_url = base_url or os.environ.get("NEBIUS_BASE_URL",
                                                   DEFAULT_BASE_URL)
        self.api_key = api_key or os.environ.get("NEBIUS_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.calls: List[dict] = []
        self.n_live = 0
        self.n_offline = 0
        self.n_invalid = 0

        self.client = None
        if not force_offline and self.api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.base_url,
                                     api_key=self.api_key,
                                     timeout=self.timeout)
            except ImportError:
                print("[mission] openai SDK missing; install it for live calls")
        if self.client is None:
            why = "no NEBIUS_API_KEY set" if not self.api_key else "forced"
            print("[mission] OFFLINE rule-based planner (%s). "
                  "Decisions are NOT Nemotron output." % why)

    # ------------------------------------------------------------------ call

    def decide(self, s: SwarmSummary) -> MissionDecision:
        """One mission decision. Falls back to rules on any failure."""
        if self.client is None:
            d = rule_based(s)
            self.n_offline += 1
            self._log(s, d)
            return d

        last_raw = ""
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": s.as_prompt()}],
                    temperature=0.2,
                    max_tokens=300,
                    response_format={"type": "json_object"},
                )
                raw = resp.choices[0].message.content or ""
                last_raw = raw
                dec = _coerce(_extract_json(raw))
                if dec is not None:
                    dec.source = self.model
                    dec.latency_ms = (time.perf_counter() - t0) * 1e3
                    dec.raw = raw
                    self.n_live += 1
                    self._log(s, dec, usage=getattr(resp, "usage", None))
                    return dec
                self.n_invalid += 1
            except Exception as exc:                      # noqa: BLE001
                last_raw = "%s: %s" % (type(exc).__name__, exc)
                if attempt == self.max_retries:
                    print("[mission] live call failed (%s); using rules" % last_raw[:120])

        d = rule_based(s)
        d.raw = last_raw
        self.n_offline += 1
        self._log(s, d, note="fell back after live failure")
        return d

    # ------------------------------------------------------------------- log

    def _log(self, s: SwarmSummary, d: MissionDecision,
             usage: Any = None, note: str = "") -> None:
        """Append a call record. This file is the submission's evidence that
        a real runtime call to Nebius Token Factory happened."""
        rec = dict(ts=time.time(), t_sim=s.t, source=d.source,
                   model=self.model, base_url=self.base_url,
                   live=d.is_live, phase_in=s.phase, phase_out=d.phase,
                   abort=d.abort, reason=d.reason,
                   latency_ms=round(d.latency_ms, 2), note=note,
                   state=asdict(s))
        if usage is not None:
            rec["usage"] = dict(
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None))
        self.calls.append(rec)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    def stats(self) -> dict:
        lat = [c["latency_ms"] for c in self.calls if c["live"]]
        return dict(live=self.n_live, offline=self.n_offline,
                    invalid_json=self.n_invalid,
                    mean_latency_ms=(sum(lat) / len(lat)) if lat else None,
                    model=self.model)


def verify_live_call(model: Optional[str] = None) -> int:
    """Prove a real Nebius Token Factory call works. Run before submitting.

        python -m mission.nemotron

    Prints the request, the raw Nemotron response and the measured latency,
    and appends the record to the call log. Exits non-zero if the call did
    not actually reach the model, so this cannot pass by accident.
    """
    m = NemotronMission(model=model)
    if m.client is None:
        print("\nFAIL: no NEBIUS_API_KEY in the environment.")
        print("  Get a key at https://tokenfactory.nebius.com and then:")
        print("    export NEBIUS_API_KEY=...        # bash")
        print("    $env:NEBIUS_API_KEY = '...'      # PowerShell")
        return 1

    s = SwarmSummary(
        t=42.0, phase="STABILISE", cable_height=3.95, peak_tension_n=6120.0,
        tension_frac_working=0.765, tension_trend=180.0,
        min_bend_radius_m=1.42, bend_margin=1.42, station_err_max_m=0.22,
        grips_held=3, grip_losses=1, mate_dist_m=0.84, mate_dwell_s=0.0,
        current_speed_est=0.38, gust_active=True,
        anomalies=["lifter 2 grip slipped at t=31 s", "surge gust rising"])
    print("model:    %s" % m.model)
    print("base_url: %s" % m.base_url)
    print("\n--- request ---\n%s" % s.as_prompt())
    d = m.decide(s)
    print("\n--- response (source=%s, %.0f ms) ---" % (d.source, d.latency_ms))
    print(d.raw or "(no raw text)")
    print("\nparsed: phase=%s abort=%s" % (d.phase, d.abort))
    print("roles:  %s" % d.roles)
    print("reason: %s" % d.reason)
    print("\nstats: %s" % m.stats())
    if not d.is_live:
        print("\nFAIL: decision came from the offline planner, not Nemotron.")
        return 1
    print("\nPASS: live Nemotron call on Nebius Token Factory. "
          "Logged to %s" % m.log_path)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(verify_live_call())
