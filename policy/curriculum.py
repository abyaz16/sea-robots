"""Training curriculum.

Without this the policy does not converge, and that is not a tuning
observation -- it is structural. At the start of training the policy is
near-random, a random policy with 2.5 kN of thrust attached to a cable
destroys the cable within a second or two, and almost every episode ends in
the same terminal penalty. The gradient carries no information about *which*
part of the behaviour was wrong.

So the environment is made easy enough that a near-random policy sometimes
survives long enough to see the reward structure, then tightened. Each stage
changes exactly one thing, so when a stage fails to progress we know what
caused it.

Progression is on *measured* success, not on a fixed step budget: a stage
advances when the policy clears its threshold, which means an easy stage
costs few steps and a hard one gets as many as it needs.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from sim.config import TaskSpec, TASK


@dataclass
class Stage:
    name: str
    profile: str
    # task relaxations applied on top of the real spec
    mate_pos_tol: float = TASK.mate_pos_tol
    mate_dwell: float = TASK.mate_dwell
    station_tol: float = TASK.station_tol
    use_sonar: bool = True
    # advance when rolling success rate exceeds this, or on step budget
    advance_success: float = 0.6
    max_steps: int = 400_000
    note: str = ""

    def task(self) -> TaskSpec:
        return TaskSpec(
            n_rov=TASK.n_rov, lift_height=TASK.lift_height,
            lift_rate=TASK.lift_rate, station_tol=self.station_tol,
            mate_pos_tol=self.mate_pos_tol,
            mate_angle_tol=TASK.mate_angle_tol, mate_dwell=self.mate_dwell,
            standoff=TASK.standoff, phase_timeout=TASK.phase_timeout)


def default_curriculum() -> List[Stage]:
    """Four stages. Each changes one thing.

    Stage 0 is the one that makes the rest possible: still water, clean
    sensing, and a mate tolerance 4x the real one. The policy only has to
    learn "lift the cable without destroying it, then close on the node".
    Sensor noise and sea state arrive only once that exists.
    """
    return [
        Stage("0-still-water", "calm", mate_pos_tol=0.32, mate_dwell=0.6,
              station_tol=0.6, use_sonar=False, advance_success=0.55,
              max_steps=500_000,
              note="no current, no sonar noise, loose tolerance"),
        Stage("1-sonar", "calm", mate_pos_tol=0.20, mate_dwell=1.0,
              station_tol=0.45, use_sonar=True, advance_success=0.5,
              max_steps=400_000,
              note="add degraded acoustic sensing"),
        Stage("2-current", "moderate", mate_pos_tol=0.12, mate_dwell=1.2,
              station_tol=0.40, use_sonar=True, advance_success=0.45,
              max_steps=500_000,
              note="add a sheared, turbulent current"),
        Stage("3-gusts", "strong", mate_pos_tol=TASK.mate_pos_tol,
              mate_dwell=TASK.mate_dwell, station_tol=TASK.station_tol,
              use_sonar=True, advance_success=1.01,   # never auto-advance
              max_steps=1_200_000,
              note="full difficulty: surge gusts and real tolerances"),
    ]


def describe(stages: List[Stage]) -> str:
    out = ["Curriculum:"]
    for i, s in enumerate(stages):
        out.append("  %d. %-14s sea=%-9s mate_tol=%.2fm dwell=%.1fs "
                   "sonar=%-5s  %s" % (
                       i, s.name, s.profile, s.mate_pos_tol, s.mate_dwell,
                       str(s.use_sonar), s.note))
    return "\n".join(out)
