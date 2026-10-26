"""The experiment: three control modes, identical physics, one variable.

    python -m eval.run_eval --runs 50
    python -m eval.run_eval --runs 8 --quick      # smoke test

Everything else in this repository is setup. This is the submission.

The only difference between the three modes is **where the control loop
runs**. Same cable, same sea states, same seeds, same task, same tolerances:

    A. teleop @ 600 ms   observations 600 ms stale, commands at 2 Hz,
                         +250 ms reaction lag       -> 0.14 Hz bandwidth
    B. teleop @ 150 ms    the same, best-case shallow water
                                                    -> 0.24 Hz bandwidth
    C. onboard autonomy   trained policy at 50 Hz, no network
                                                    -> 7.96 Hz bandwidth

Seeds are matched across modes: run *i* of a given sea state faces the
identical current realisation in every mode. A difference between modes is
therefore attributable to the controller and not to luck, and the paired
structure also allows a paired test rather than an unpaired one.

`pd_ideal` (zero latency, perfect state) is also run, not as a competitor but
as an upper bound: it says how much of the remaining failure is the task
being hard rather than the loop being slow.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

from baselines.teleop import MODES, TeleopController
from sim.config import EVAL_PROFILES, PROFILES, TASK
from sim.env import RepairEnv

MODE_ORDER = ["teleop_600", "teleop_150", "autonomy", "pd_ideal"]
MODE_LABEL = {
    "teleop_600": "A. Teleop 600ms",
    "teleop_150": "B. Teleop 150ms",
    "autonomy": "C. Onboard autonomy",
    "pd_ideal": "   (oracle: 0ms, full state)",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leviathan evaluation")
    p.add_argument("--runs", type=int, default=50, help="runs per condition")
    p.add_argument("--policy", type=str, default="data/runs/ppo/policy.pt")
    p.add_argument("--out", type=str, default="data/runs/eval")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seas", type=str, default=",".join(EVAL_PROFILES))
    p.add_argument("--modes", type=str, default=",".join(MODE_ORDER))
    p.add_argument("--max-seconds", type=float, default=60.0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--quick", action="store_true",
                   help="fewer runs, two seas: for wiring checks only")
    p.add_argument("--record", type=str, default="",
                   help="sea state to record trajectories for (viz)")
    return p.parse_args()


def load_policy(path: str, obs_dim: int, device):
    """Load a trained policy, or return None if there is not one yet."""
    p = Path(path)
    if not p.exists():
        return None
    from policy.nets import ActorCritic
    ck = torch.load(p, map_location=device, weights_only=False)
    if int(ck.get("obs_dim", obs_dim)) != obs_dim:
        print("  !! policy obs_dim %s != env obs_dim %s; skipping autonomy"
              % (ck.get("obs_dim"), obs_dim))
        return None
    net = ActorCritic(obs_dim).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    print("  loaded policy: step %s, stage %s" % (ck.get("step"), ck.get("stage")))
    return net


class Recorder:
    """Captures a trajectory for the replay viewer.

    Downsampled to 20 Hz and to every other cable node: the viewer needs
    smooth motion, not every solver state, and a 60 s run at full rate would
    be a 40 MB JSON nobody can load in a browser.
    """

    def __init__(self, env: RepairEnv, stride: int = 3, node_stride: int = 2):
        self.env = env
        self.stride = stride
        self.node_stride = node_stride
        self.frames: List[dict] = []
        self.k = 0

    def capture(self, info: dict) -> None:
        if self.k % self.stride == 0:
            e = self.env
            r = lambda t: [round(float(v), 3) for v in t.flatten().tolist()]
            self.frames.append(dict(
                t=round(float(e.t[0]), 2),
                phase=int(e.phase[0]),
                cable=r(e.cable.x[0, ::self.node_stride]),
                rov=r(e.swarm.pos[0]),
                yaw=r(e.swarm.yaw[0]),
                tension=round(float(info["peak_tension"][0]), 1),
                bend=round(float(info["min_radius"][0]), 3),
                mate=round(float(info["mate_dist"][0]), 3),
            ))
        self.k += 1

    def dump(self, path: Path, mode: str, sea: str, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(
            mode=mode, sea=sea, dt=self.stride * self.env.sim.control_dt,
            n_nodes=len(self.frames[0]["cable"]) // 3 if self.frames else 0,
            n_rov=self.env.n_rov, meta=meta, frames=self.frames), indent=0))


def run_condition(mode: str, sea: str, runs: int, args, device,
                  policy=None, record_to: Optional[Path] = None) -> List[dict]:
    """One mode in one sea state, `runs` episodes in parallel."""
    env = RepairEnv(batch=runs, profile=sea, device=device, seed=hash(sea) % 10000,
                    task=TASK, use_sonar=True)
    ctrl = None
    if mode != "autonomy":
        ctrl = TeleopController(env, MODES[mode])
    obs = env.reset()
    if ctrl is not None:
        ctrl.reset()

    rec = Recorder(env) if record_to is not None else None
    steps = int(args.max_seconds * env.sim.control_hz)
    for _ in range(steps):
        if mode == "autonomy":
            with torch.no_grad():
                act, _, _ = policy.act(obs, deterministic=True)
        else:
            act = ctrl.act()
        obs, _, done, info = env.step(act)
        if rec is not None:
            rec.capture(info)
        if bool(done.all()):
            break

    m = env.episode_metrics()
    rows = []
    for i in range(runs):
        rows.append(dict(
            mode=mode, sea=sea, run=i,
            held_out=bool(PROFILES[sea].held_out),
            success=bool(m["success"][i] > 0.5),
            peak_tension_frac=float(m["peak_tension_frac"][i]),
            peak_tension_n=float(m["peak_tension_n"][i]),
            min_bend_radius=float(m["min_bend_radius"][i]),
            over_tension_events=float(m["over_tension_events"][i]),
            bend_events=float(m["bend_events"][i]),
            station_rms=float(m["station_rms"][i]),
            time_s=float(m["time_s"][i]),
            energy=float(m["energy"][i]),
            fail_reason=int(m["fail_reason"][i]),
        ))
    if rec is not None and rec.frames:
        rec.dump(record_to, mode, sea,
                 meta=dict(success=rows[0]["success"],
                           peak_tension_frac=rows[0]["peak_tension_frac"],
                           min_bend_radius=rows[0]["min_bend_radius"]))
    return rows


def main() -> int:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.quick:
        args.runs = min(args.runs, 8)
        seas = ["moderate", "surge"]
    else:
        seas = [s for s in args.seas.split(",") if s]
    modes = [m for m in args.modes.split(",") if m]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    probe = RepairEnv(batch=2, device=device)
    policy = load_policy(args.policy, probe.obs_dim, device)
    if policy is None and "autonomy" in modes:
        print("  no usable policy at %s -- autonomy mode will be skipped"
              % args.policy)
        modes = [m for m in modes if m != "autonomy"]

    print("\nevaluating %d mode(s) x %d sea(s) x %d runs = %d episodes"
          % (len(modes), len(seas), args.runs, len(modes) * len(seas) * args.runs))
    print("device %s | max %.0f s per episode\n" % (device, args.max_seconds))

    rows: List[dict] = []
    t0 = time.perf_counter()
    for mode in modes:
        for sea in seas:
            ts = time.perf_counter()
            rec_path = None
            if args.record and sea == args.record:
                rec_path = out / "traj_%s_%s.json" % (mode, sea)
            r = run_condition(mode, sea, args.runs, args, device, policy,
                              record_to=rec_path)
            rows.extend(r)
            ok = sum(x["success"] for x in r) / len(r)
            print("  %-12s %-9s success %5.1f%%  peakT %5.1f%%  minR %.2f m  "
                  "(%.0f s)" % (
                      mode, sea, 100 * ok,
                      100 * sum(x["peak_tension_frac"] for x in r) / len(r),
                      sum(x["min_bend_radius"] for x in r) / len(r),
                      time.perf_counter() - ts))

    (out / "results.json").write_text(json.dumps(rows, indent=1))
    meta = dict(runs=args.runs, seas=seas, modes=modes,
                max_seconds=args.max_seconds,
                mate_pos_tol=TASK.mate_pos_tol, mate_dwell=TASK.mate_dwell,
                policy=args.policy, device=str(device),
                wall_minutes=(time.perf_counter() - t0) / 60.0,
                corner_hz={m: MODES[m].corner_hz for m in MODES})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("\nwrote %d rows to %s in %.1f min"
          % (len(rows), out / "results.json", meta["wall_minutes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
