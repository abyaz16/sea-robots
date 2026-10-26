"""Physical constants and tunables for Project Leviathan.

Every number here is either (a) taken from public subsea-industry data, or
(b) a documented reduced-order modelling choice. They live in one module on
purpose: the "honest limits" section of the submission is generated from this
file, so a reviewer can audit every assumption in one place.

World frame: x east, y north, z UP from the seabed plane (z=0). SI units.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import math

# ---------------------------------------------------------------- environment

RHO_WATER = 1025.0   # kg/m^3, seawater at depth
GRAVITY = 9.80665    # m/s^2


# --------------------------------------------------------------------- cable

@dataclass(frozen=True)
class CableSpec:
    """Armoured submarine telecom cable, repeater-section class.

    Linear density and diameter are representative of double-armoured cable.
    Break tension and minimum bend radius are the two hard limits the policy
    must respect; both are enforced as episode-terminating failures.
    """
    n_nodes: int = 41                 # 40 segments
    seg_len: float = 0.5              # m  -> 20 m of modelled cable
    lin_density: float = 3.5          # kg/m, in air
    diameter: float = 0.050           # m, over armour
    break_tension: float = 30.0e3     # N   -> HARD FAIL if exceeded
    min_bend_radius: float = 1.0      # m   -> HARD FAIL if violated

    # Maximum permissible working tension. Industry rates cables for both an
    # ultimate break load and a far lower working load; exceeding the working
    # load strains the fibre and raises attenuation permanently, long before
    # anything snaps. Real cable-handling procedures are written against this
    # number, not against break strength, so it is what we score against.
    working_tension: float = 8.0e3    # N, ~27% of break load

    # XPBD compliances (inverse stiffness). alpha = 0 is a rigid constraint.
    axial_compliance: float = 1.0e-9  # near-inextensible
    bend_compliance: float = 3.0e-6   # tuned so the cable behaves like a
                                      # stiff hose, not a rope (see docs)

    # Hydrodynamics, per node, Morison-style
    cd_normal: float = 1.2            # cylinder cross-flow
    cd_tangent: float = 0.01          # skin friction along the axis
    ca_normal: float = 1.0            # added-mass coefficient, cross-flow
    ca_tangent: float = 0.0           # negligible along the axis

    # Vortex-induced vibration (cross-flow lift from vortex shedding).
    #
    # This is the *fast* disturbance in the problem, and it is the reason
    # loop latency matters at all. Shedding frequency f = St*u/D: at 0.05 m
    # diameter and 0.3 m/s that is ~1.2 Hz, i.e. right in the band a 600 ms
    # control loop cannot track, whereas the turbulent current sits two
    # decades lower at 0.04-0.33 Hz. A slow-current-only model would make
    # teleop look far better than it is.
    #
    # Cl = 0.7 is a mid-range cross-flow lift coefficient for a cylinder near
    # lock-in. We do not model lock-in amplification itself (which would make
    # VIV worse), so this is a conservative choice.
    viv_enabled: bool = True
    viv_strouhal: float = 0.20
    viv_cl: float = 0.7
    viv_corr_length: float = 2.0      # m, spanwise coherence of shedding;
                                      # shorter = more incoherent = weaker
                                      # net load on the span

    @property
    def node_mass(self) -> float:
        return self.lin_density * self.seg_len

    @property
    def node_volume(self) -> float:
        return math.pi * 0.25 * self.diameter ** 2 * self.seg_len

    @property
    def submerged_weight_per_m(self) -> float:
        """Net downward force per metre in seawater. ~14.6 N/m as configured."""
        displaced = RHO_WATER * math.pi * 0.25 * self.diameter ** 2
        return (self.lin_density - displaced) * GRAVITY

    @property
    def max_turn_angle(self) -> float:
        """Per-joint turning angle at which bend radius hits the limit.

        A discrete rod of segment length L bent to radius R turns by
        phi = 2*asin(L / 2R) per joint. R = R_min gives the allowed maximum.
        """
        return 2.0 * math.asin(min(1.0, self.seg_len / (2.0 * self.min_bend_radius)))


# ----------------------------------------------------------------------- ROV

@dataclass(frozen=True)
class ROVSpec:
    """Work-class ROV, 4-DOF control (surge, sway, heave, yaw).

    Roll and pitch are assumed actively stabilised by trim and are not
    modelled -- a reduced-order choice, stated in the limits section.

    The policy emits a 4-vector wrench demand in [-1, 1]. That demand is
    allocated across a 6-thruster BlueROV-style layout (4 vectored horizontal
    + 2 vertical), so saturating any single physical thruster clips the
    achievable wrench. The policy therefore has to learn the real actuation
    envelope, not an idealised decoupled one.
    """
    mass: float = 1500.0              # kg, in air
    volume: float = 1.47              # m^3 -> ~neutrally buoyant in seawater
    inertia_yaw: float = 900.0        # kg m^2

    max_force_xy: float = 2500.0      # N, surge / sway
    max_force_z: float = 2000.0       # N, heave
    max_torque_yaw: float = 1500.0    # N m

    thrust_rate_limit: float = 4.0    # per second, normalised demand slew
    thrust_tau: float = 0.15          # s, first-order thruster lag

    # Translational drag, F = -0.5 rho Cd A |v| v
    cd_xy: float = 0.9
    cd_z: float = 1.1
    area_xy: float = 1.6              # m^2
    area_z: float = 2.4               # m^2
    ca_xy: float = 0.9                # added mass coefficients
    ca_z: float = 1.2
    damp_yaw: float = 1800.0          # N m s^2 / rad^2, quadratic yaw damping

    # Gripper: a compliant spring-damper tie between the ROV and its cable
    # node, standing in for a manipulator arm plus a short soft lift line.
    #
    # Stiffness is set from the load it must carry, not picked for feel: each
    # grip holds roughly 220 N of submerged cable, so 4 kN/m gives ~55 mm of
    # working deflection -- compliant enough to absorb a snatch load, stiff
    # enough to actually control the grip point.
    #
    # Damping is bounded by numerical stability, not by preference. Gripper
    # forces enter the cable once per 120 Hz physics step, so an explicit
    # damper needs c*dt/m < 2; with a 2.75 kg effective node mass that caps c
    # near 660 N s/m. 200 N s/m keeps a comfortable margin.
    grip_stiffness: float = 4.0e3     # N/m
    grip_damping: float = 2.0e2       # N s/m
    grip_max_force: float = 1.0e4     # N, the latch lets go above this
                                      # (> 4x single-ROV thrust, so losing a
                                      # grip is a surge event, not routine)

    @property
    def net_buoyancy(self) -> float:
        """Upward force (N). Positive = buoyant. Slightly positive by design,
        as real ROVs are, so they surface on power loss."""
        return (RHO_WATER * self.volume - self.mass) * GRAVITY

    @property
    def eff_mass_xy(self) -> float:
        return self.mass + self.ca_xy * RHO_WATER * self.volume

    @property
    def eff_mass_z(self) -> float:
        return self.mass + self.ca_z * RHO_WATER * self.volume


# ------------------------------------------------------------------- current

@dataclass(frozen=True)
class CurrentProfile:
    """A named sea state. `u_ref` is the mean speed at `z_ref` above seabed."""
    name: str
    u_ref: float                      # m/s
    z_ref: float = 10.0               # m
    turb_intensity: float = 0.08      # fraction of u_ref
    gust_amp: float = 0.0             # multiple of u_ref added at gust peak
    gust_period: Tuple[float, float] = (20.0, 40.0)   # s, sampled per episode
    gust_width: float = 2.5           # s, gaussian sigma of the surge
    held_out: bool = False            # not seen during training


# 1 knot = 0.5144 m/s. Deep-ocean bottom currents are typically 0.05-0.25 m/s;
# 0.5 m/s+ is a genuinely difficult working condition and is where real
# operations get suspended.
PROFILES: Dict[str, CurrentProfile] = {
    # ---- seen during training
    "calm":     CurrentProfile("calm",     u_ref=0.05, turb_intensity=0.05),
    "moderate": CurrentProfile("moderate", u_ref=0.20, turb_intensity=0.10),
    "strong":   CurrentProfile("strong",   u_ref=0.40, turb_intensity=0.14,
                               gust_amp=0.8),
    # ---- held out: never trained on, used to test generalisation
    "shear":    CurrentProfile("shear",    u_ref=0.30, z_ref=4.0,
                               turb_intensity=0.18, held_out=True),
    "eddy":     CurrentProfile("eddy",     u_ref=0.28, turb_intensity=0.30,
                               gust_amp=0.5, held_out=True),
    # ---- the money condition: a surge no 600 ms teleop loop can track
    "surge":    CurrentProfile("surge",    u_ref=0.35, turb_intensity=0.12,
                               gust_amp=2.0, gust_period=(12.0, 20.0),
                               gust_width=1.6, held_out=True),
}

TRAIN_PROFILES = [k for k, v in PROFILES.items() if not v.held_out]
EVAL_PROFILES = list(PROFILES.keys())


# ------------------------------------------------------------------- solver

@dataclass(frozen=True)
class SonarSpec:
    """Acoustic sensing error model.

    Dropout is a two-state Markov chain, so losses arrive in bursts: with
    these rates a dropout lasts ~8 control steps (160 ms) on average and
    occurs a few times a minute per target. Independent per-step dropout
    would be trivially filterable and would flatter the policy.
    """
    range_noise_frac: float = 0.02      # 2% of range, 1-sigma
    bearing_noise_rad: float = 0.012    # ~0.7 deg, 1-sigma
    dropout_enter: float = 0.004        # per control step, per target
    dropout_exit: float = 0.12          # per control step -> ~8-step bursts
    stale_steps: int = 1                # control steps of processing latency


@dataclass(frozen=True)
class TaskSpec:
    """The repair task: geometry, tolerances and phase structure."""
    n_rov: int = 4                      # 3 lifters + 1 connector vehicle
    lift_height: float = 4.0            # m above the seabed
    lift_rate: float = 0.35             # m/s, commanded ramp
    station_tol: float = 0.35           # m, lifter station-keeping tolerance
    mate_pos_tol: float = 0.08          # m, connector alignment tolerance
    mate_angle_tol: float = 0.15        # rad
    mate_dwell: float = 1.5             # s inside tolerance to count as mated
    standoff: float = 6.0               # m, initial ROV standoff from cable
    phase_timeout: tuple = (25.0, 20.0, 45.0)   # LIFT, STABILISE, MATE


@dataclass(frozen=True)
class SimSpec:
    physics_hz: float = 150.0
    control_hz: float = 50.0
    mission_hz: float = 0.25          # Nemotron, every 4 s
    xpbd_substeps: int = 6            # per physics step
    max_velocity: float = 25.0        # m/s clamp, blow-up guard
    episode_seconds: float = 95.0

    @property
    def dt(self) -> float:
        return 1.0 / self.physics_hz

    @property
    def sub_dt(self) -> float:
        return self.dt / self.xpbd_substeps

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def phys_per_control(self) -> int:
        return int(round(self.physics_hz / self.control_hz))


CABLE = CableSpec()
ROV = ROVSpec()
SIM = SimSpec()
SONAR = SonarSpec()
TASK = TaskSpec()


def summary() -> str:
    """Human-readable dump, used when generating docs/SUBMISSION.md."""
    lines = [
        "Cable:  %d segments x %.2f m = %.1f m modelled" % (
            CABLE.n_nodes - 1, CABLE.seg_len,
            (CABLE.n_nodes - 1) * CABLE.seg_len),
        "        %.1f kg/m in air, %.1f N/m submerged weight" % (
            CABLE.lin_density, CABLE.submerged_weight_per_m),
        "        break %.0f kN, min bend radius %.2f m (max joint turn %.1f deg)" % (
            CABLE.break_tension / 1e3, CABLE.min_bend_radius,
            math.degrees(CABLE.max_turn_angle)),
        "ROV:    %.0f kg, net buoyancy %+.0f N, thrust %.0f/%.0f/%.0f N + %.0f Nm" % (
            ROV.mass, ROV.net_buoyancy, ROV.max_force_xy, ROV.max_force_xy,
            ROV.max_force_z, ROV.max_torque_yaw),
        "Solver: physics %.0f Hz x %d XPBD substeps, control %.0f Hz, mission %.2f Hz" % (
            SIM.physics_hz, SIM.xpbd_substeps, SIM.control_hz, SIM.mission_hz),
        "Seas:   train %s | held out %s" % (
            ", ".join(TRAIN_PROFILES),
            ", ".join(k for k in PROFILES if PROFILES[k].held_out)),
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
