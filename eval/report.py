"""Turn eval results into the submission's table and figures.

    python -m eval.report

Writes `docs/RESULTS.md`, plus `docs/fig_degradation.png` and
`docs/fig_training.png` when matplotlib is available.

Reporting choices worth stating
-------------------------------
* **Peak tension is reported against the working limit, not break load.**
  Cable handling procedures are written against maximum permissible working
  tension, because exceeding it strains the fibre permanently. Reporting
  against the 30 kN break load would make every number look comfortable and
  would be the wrong engineering comparison.
* **Held-out sea states are marked and reported separately.** A result that
  only holds on trained conditions is a memorisation result.
* **Runs are paired across modes by seed**, so the mode comparison uses a
  paired test. Unpaired would throw away the matching and widen the
  intervals for no reason.
* **Success-rate intervals are Wilson intervals**, not normal approximations,
  because at 50 runs and rates near 0 or 1 the normal approximation produces
  intervals that extend past 0% or 100%.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from eval.run_eval import MODE_LABEL, MODE_ORDER
from sim.config import CABLE, PROFILES, TASK

RESULTS = Path("data/runs/eval/results.json")
META = Path("data/runs/eval/meta.json")
HISTORY = Path("data/runs/ppo/history.json")
OUT = Path("docs/RESULTS.md")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - half), min(1.0, c + half))


def mean(xs: List[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def paired_test(a: List[bool], b: List[bool]) -> dict:
    """McNemar-style paired comparison of two success vectors.

    Counts only the runs where the two modes disagreed -- which is the whole
    point of pairing. With matched seeds, a run both modes pass (or both
    fail) carries no information about which is better.
    """
    n = min(len(a), len(b))
    a_only = sum(1 for i in range(n) if a[i] and not b[i])
    b_only = sum(1 for i in range(n) if b[i] and not a[i])
    disc = a_only + b_only
    if disc == 0:
        return dict(a_only=0, b_only=0, discordant=0, p=1.0)
    # exact two-sided binomial test at p=0.5 on the discordant pairs
    tail = sum(math.comb(disc, i) for i in range(0, min(a_only, b_only) + 1))
    p = min(1.0, 2.0 * tail / (2 ** disc))
    return dict(a_only=a_only, b_only=b_only, discordant=disc, p=p)


def load() -> tuple:
    if not RESULTS.exists():
        raise SystemExit("no results at %s -- run `python -m eval.run_eval` first"
                         % RESULTS)
    rows = json.loads(RESULTS.read_text())
    meta = json.loads(META.read_text()) if META.exists() else {}
    return rows, meta


def table_overall(rows: List[dict]) -> str:
    by_mode: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_mode[r["mode"]].append(r)

    out = ["| Mode | Bandwidth | Mate OK | 95% CI | Peak tension "
           "(% working) | Over-tension /run | Bend viol. /run | "
           "Station RMS (m) | Time (s) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for m in MODE_ORDER:
        rs = by_mode.get(m)
        if not rs:
            continue
        k = sum(r["success"] for r in rs)
        p, lo, hi = wilson(k, len(rs))
        bw = CORNER.get(m)
        out.append("| %s | %s | **%.0f%%** | %.0f–%.0f%% | %.0f%% | %.1f | "
                   "%.2f | %.3f | %.0f |" % (
                       MODE_LABEL[m], ("%.2f Hz" % bw) if bw else "-",
                       100 * p, 100 * lo, 100 * hi,
                       100 * mean([r["peak_tension_frac"] for r in rs]),
                       mean([r["over_tension_events"] for r in rs]),
                       mean([r["bend_events"] for r in rs]),
                       mean([r["station_rms"] for r in rs]),
                       mean([r["time_s"] for r in rs])))
    return "\n".join(out)


def table_by_sea(rows: List[dict]) -> str:
    seas = []
    for r in rows:
        if r["sea"] not in seas:
            seas.append(r["sea"])
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in rows)]

    head = "| Sea state | u_ref | Seen in training | " + \
        " | ".join(MODE_LABEL[m].strip() for m in modes) + " |"
    out = [head, "|---" * (3 + len(modes)) + "|"]
    for sea in seas:
        cells = []
        for m in modes:
            rs = [r for r in rows if r["sea"] == sea and r["mode"] == m]
            if not rs:
                cells.append("-")
                continue
            p, lo, hi = wilson(sum(r["success"] for r in rs), len(rs))
            cells.append("%.0f%%" % (100 * p))
        pr = PROFILES[sea]
        out.append("| %s | %.2f m/s | %s | %s |" % (
            sea, pr.u_ref, "no — held out" if pr.held_out else "yes",
            " | ".join(cells)))
    return "\n".join(out)


def failure_breakdown(rows: List[dict]) -> str:
    names = {0: "none (succeeded)", 1: "cable broke", 2: "bend radius violated",
             3: "timed out", 4: "grip lost"}
    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in rows)]
    out = ["| Mode | " + " | ".join(names[i] for i in sorted(names)) + " |",
           "|---" * (1 + len(names)) + "|"]
    for m in modes:
        rs = [r for r in rows if r["mode"] == m]
        counts = defaultdict(int)
        for r in rs:
            counts[r["fail_reason"]] += 1
        out.append("| %s | %s |" % (
            MODE_LABEL[m].strip(),
            " | ".join("%d" % counts[i] for i in sorted(names))))
    return "\n".join(out)


def degradation_figure(rows: List[dict]) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    modes = [m for m in MODE_ORDER if any(r["mode"] == m for r in rows)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    style = {"teleop_600": ("#d1495b", "o", "-"),
             "teleop_150": ("#edae49", "s", "-"),
             "autonomy": ("#1b998b", "D", "-"),
             "pd_ideal": ("#8d99ae", ".", ":")}
    for m in modes:
        pts = []
        for sea in {r["sea"] for r in rows if r["mode"] == m}:
            rs = [r for r in rows if r["sea"] == sea and r["mode"] == m]
            p, lo, hi = wilson(sum(r["success"] for r in rs), len(rs))
            pts.append((PROFILES[sea].u_ref, 100 * p, 100 * (p - lo),
                        100 * (hi - p)))
        pts.sort()
        x = [a for a, _, _, _ in pts]
        y = [b for _, b, _, _ in pts]
        el = [c for _, _, c, _ in pts]
        eu = [d for _, _, _, d in pts]
        c, mk, ls = style.get(m, ("#555", "o", "-"))
        ax.errorbar(x, y, yerr=[el, eu], color=c, marker=mk, ls=ls,
                    capsize=3, lw=2, ms=6, label=MODE_LABEL[m].strip())
    ax.set_xlabel("Mean current speed at 10 m above seabed (m/s)")
    ax.set_ylabel("Mate success rate (%)")
    ax.set_title("Autonomy degrades gracefully; teleop falls off a cliff")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    Path("docs").mkdir(exist_ok=True)
    fig.savefig("docs/fig_degradation.png")
    plt.close(fig)
    return True


def training_figure() -> bool:
    if not HISTORY.exists():
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    h = json.loads(HISTORY.read_text())
    if not h:
        return False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.2), dpi=160,
                                 sharex=True)
    step = [d["step"] / 1e6 for d in h]
    a1.plot(step, [d["ep_return"] for d in h], color="#1b998b", lw=1.8)
    a1.set_ylabel("episode return")
    a1.grid(alpha=0.25)
    a2.plot(step, [100 * d["success"] for d in h], color="#d1495b", lw=1.8)
    a2.set_ylabel("mate success (%)")
    a2.set_xlabel("environment steps (millions)")
    a2.grid(alpha=0.25)
    # mark curriculum stage changes: the visible discontinuities
    last = None
    for d in h:
        if last is not None and d["stage"] != last:
            for ax in (a1, a2):
                ax.axvline(d["step"] / 1e6, color="#555", ls="--", lw=0.9,
                           alpha=0.7)
            a1.annotate(d.get("stage_name", str(d["stage"])),
                        (d["step"] / 1e6, a1.get_ylim()[1]),
                        fontsize=7, rotation=90, va="top", ha="right")
        last = d["stage"]
    a1.set_title("Training on Nebius AI Cloud — curriculum stages marked")
    fig.tight_layout()
    fig.savefig("docs/fig_training.png")
    plt.close(fig)
    return True


CORNER: Dict[str, float] = {}


def main() -> int:
    global CORNER
    rows, meta = load()
    CORNER = meta.get("corner_hz", {})

    seen = [r for r in rows if not r["held_out"]]
    held = [r for r in rows if r["held_out"]]

    parts = ["# Results", "",
             "Three control modes, identical physics, identical paired seeds. "
             "The only variable is **where the control loop runs**.", ""]
    if meta:
        parts += ["%d runs per condition, %d sea states, %d modes. "
                  "Mate tolerance %.0f cm held for %.1f s. "
                  "Evaluated on %s in %.1f min." % (
                      meta.get("runs", 0), len(meta.get("seas", [])),
                      len(meta.get("modes", [])), 100 * TASK.mate_pos_tol,
                      TASK.mate_dwell, meta.get("device", "?"),
                      meta.get("wall_minutes", 0.0)), ""]

    parts += ["## Overall", "", table_overall(rows), "",
              "Peak tension is a fraction of the **%.0f kN working limit** "
              "(%.0f kN break load), because cable-handling procedure is "
              "written against working tension: exceeding it raises fibre "
              "attenuation permanently, long before anything snaps." % (
                  CABLE.working_tension / 1e3, CABLE.break_tension / 1e3), "",
              "## By sea state", "", table_by_sea(rows), "",
              "## How runs failed", "", failure_breakdown(rows), ""]

    # paired comparison, autonomy vs each teleop mode
    def vec(mode: str, subset: List[dict]) -> List[bool]:
        rs = [r for r in subset if r["mode"] == mode]
        rs.sort(key=lambda r: (r["sea"], r["run"]))
        return [r["success"] for r in rs]

    if any(r["mode"] == "autonomy" for r in rows):
        parts += ["## Paired comparison", "",
                  "Matched seeds, so each run faces the same current "
                  "realisation in every mode. Counts are discordant pairs "
                  "(runs where exactly one mode succeeded); *p* is an exact "
                  "two-sided binomial test.", "",
                  "| Comparison | Autonomy only | Teleop only | Discordant | p |",
                  "|---|---|---|---|---|"]
        for m in ("teleop_600", "teleop_150"):
            if not any(r["mode"] == m for r in rows):
                continue
            t = paired_test(vec("autonomy", rows), vec(m, rows))
            parts.append("| autonomy vs %s | %d | %d | %d | %.4g |" % (
                m, t["a_only"], t["b_only"], t["discordant"], t["p"]))
        parts.append("")

    if held:
        for label, subset in (("trained sea states", seen),
                              ("held-out sea states", held)):
            if not subset:
                continue
            parts += ["### Generalisation — %s" % label, ""]
            for m in MODE_ORDER:
                rs = [r for r in subset if r["mode"] == m]
                if not rs:
                    continue
                p, lo, hi = wilson(sum(r["success"] for r in rs), len(rs))
                parts.append("- %s: **%.0f%%** (%.0f–%.0f%%, n=%d)" % (
                    MODE_LABEL[m].strip(), 100 * p, 100 * lo, 100 * hi, len(rs)))
            parts.append("")

    figs = []
    if degradation_figure(rows):
        figs.append("![Degradation curve](fig_degradation.png)")
    if training_figure():
        figs.append("![Training curves](fig_training.png)")
    if figs:
        parts += ["## Figures", ""] + figs + [""]

    parts += ["## Reading this honestly", "",
              "The teleop baseline is a **delayed PD controller, not a "
              "human** — see [LIMITS.md](LIMITS.md) for exactly what it does "
              "and does not model, and [LATENCY.md](LATENCY.md) for the "
              "bandwidth derivation and the measurement that could have "
              "falsified the whole claim.", ""]

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote %s" % OUT)
    print("\n".join(parts[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
