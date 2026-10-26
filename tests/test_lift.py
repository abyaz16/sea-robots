"""Smoke test for the ROV <-> cable gripper coupling.

Two ROVs grip the cable a third of the way in from each end and lift it off
the seabed with a crude PD controller. This is not the task and the PD is not
the policy -- it exists to prove that the one interface Day 2 depends on
actually closes: cable node forces in, ROV reaction forces out, momentum
conserved, nothing exploding.

What it checks:
  * the lift happens at all (cable comes off the bed)
  * tension stays below the working limit under a hand controller
  * bend radius at the lift points stays legal
  * forces are equal and opposite (no momentum injected by the gripper)

Run: python -m tests.test_lift
"""

import sys

import torch

from sim.cable import Cable
from sim.config import CABLE, ROV, SIM, PROFILES
from sim.rov import ROVSwarm
from sim.water import CurrentField


def main(sea: str = "moderate", seconds: float = 12.0, seed: int = 0) -> int:
    device = "cpu"
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    B, K = 2, 2

    cable = Cable(CABLE, B, device=device)
    cable.reset_seabed()
    field = CurrentField(PROFILES[sea], B, device=device, generator=gen)
    swarm = ROVSwarm(ROV, B, K, device=device)

    # Grip a third in from each end, and start the vehicles just above.
    nodes = [CABLE.n_nodes // 3, 2 * CABLE.n_nodes // 3]
    start = torch.stack([cable.x[:, n] for n in nodes], dim=1).clone()
    start[..., 2] += 0.8
    swarm.reset(start)                 # reset() clears grips, so grip after
    for k, n in enumerate(nodes):
        swarm.grip_node[:, k] = n

    # Target: lift the grip points to 4 m above the bed, holding xy.
    # Ramped at 0.4 m/s, because a real cable lift is slow and a step command
    # would be testing the gripper's slip limit rather than the coupling.
    target = start.clone()
    z0 = start[..., 2].clone()
    lift_rate, lift_to = 0.4, 4.0

    steps = int(seconds * SIM.physics_hz)
    t = torch.zeros(B, device=device)
    worst = dict(tension=0.0, min_r=1e9, imbalance=0.0)
    lift_height = 0.0

    for i in range(steps):
        f_cable, f_rov = swarm.gripper_forces(cable.x, cable.v)

        # Newton's third law: what the gripper puts into the cable must come
        # back out of the vehicles. If this drifts, the sim is manufacturing
        # momentum and every result downstream is void.
        imbalance = float((f_cable.sum(dim=1) + f_rov.sum(dim=1)).norm(dim=-1).max())
        worst["imbalance"] = max(worst["imbalance"], imbalance)

        target[..., 2] = torch.minimum(z0 + lift_rate * float(t[0]),
                                       torch.full_like(z0, lift_to))

        # crude PD on position, plus a feedforward hold for cable weight
        err = target - swarm.pos
        dem = torch.zeros(B, K, 4, device=device)
        dem[..., 0] = (0.8 * err[..., 0] - 1.2 * swarm.vel[..., 0]).clamp(-1, 1)
        dem[..., 1] = (0.8 * err[..., 1] - 1.2 * swarm.vel[..., 1]).clamp(-1, 1)
        dem[..., 2] = (0.9 * err[..., 2] - 1.5 * swarm.vel[..., 2] + 0.25).clamp(-1, 1)

        swarm.step(SIM.dt, dem, field, t, grip_force=f_rov)
        cable.step(SIM.dt, field, t, ext_force=f_cable,
                   substeps=SIM.xpbd_substeps, max_velocity=SIM.max_velocity)
        t = t + SIM.dt

        if i % 20 == 0:
            d = cable.diagnostics()
            if not bool(d["finite"].all()):
                print("FAILED: non-finite state at step %d" % i)
                return 1
            worst["tension"] = max(worst["tension"], float(d["peak_tension"].max()))
            worst["min_r"] = min(worst["min_r"], float(d["min_bend_radius"].min()))
            lift_height = max(lift_height, float(cable.x[..., 2].max()))

    held = int((swarm.grip_node >= 0).sum())
    print("sea=%s  %.0f s" % (sea, seconds))
    print("  lift height        %.2f m   (target 4.0, from the seabed)" % lift_height)
    print("  peak tension       %.0f N   (%.1f%% of %.0f N working limit)" % (
        worst["tension"], 100.0 * worst["tension"] / CABLE.working_tension,
        CABLE.working_tension))
    print("  min bend radius    %.2f m   (limit %.2f m)" % (
        worst["min_r"], CABLE.min_bend_radius))
    print("  grips still held   %d / %d" % (held, B * K))
    print("  max force residual %.2e N   (Newton's third law)" % worst["imbalance"])

    fails = []
    if lift_height < 1.5:
        fails.append("cable never came off the seabed (%.2f m)" % lift_height)
    if worst["tension"] > CABLE.working_tension:
        fails.append("exceeded working tension (%.0f N)" % worst["tension"])
    if worst["min_r"] < CABLE.min_bend_radius:
        fails.append("bend radius violated (%.2f m)" % worst["min_r"])
    if worst["imbalance"] > 1e-3:
        fails.append("gripper injects momentum (residual %.2e N)" % worst["imbalance"])

    print()
    if fails:
        print("LIFT TEST FAILED")
        for f in fails:
            print("  - %s" % f)
        return 1
    print("LIFT TEST PASSED: gripper coupling closes, two ROVs lift the "
          "cable within both limits.")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(sea=a[0] if a else "moderate",
                  seconds=float(a[1]) if len(a) > 1 else 12.0,
                  seed=int(a[2]) if len(a) > 2 else 0))
