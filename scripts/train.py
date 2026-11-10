"""PPO training on MJX (GPU). One run at a time -- do not launch two concurrently.

    scripts/gpu_python.sh scripts/train.py --name stand_A8_s0 --model baseline_A_8dof --task stand \
        --steps 100_000_000 [--dr] [--push 0.3] [--seed 0] [--init runs/<other>/params.pkl]

Writes runs/<name>/{config.json, progress.jsonl, params_*.pkl, params.pkl, done.json}.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _install_jax_shims():
    """Brax 0.14 still calls pmap-era helpers that JAX 0.12 removed; single-host equivalents."""
    def put_replicated(x, devices):
        return jax.tree.map(lambda a: jax.device_put(jnp.stack([jnp.asarray(a)] * len(devices)), devices[0])
                            if len(devices) == 1 else jax.device_put_sharded([a] * len(devices), devices), x)

    def put_sharded(xs, devices):
        return jax.tree.map(lambda *a: jax.device_put(jnp.stack(a), devices[0]), *xs)

    for name, fn in (("device_put_replicated", put_replicated), ("device_put_sharded", put_sharded)):
        try:
            getattr(jax, name)
        except AttributeError:
            setattr(jax, name, fn)


_install_jax_shims()
from brax.training.agents.ppo import networks as ppo_networks  # noqa: E402
from brax.training.agents.ppo import train as ppo  # noqa: E402

from slender_biped.mjx_env import EnvConfig, SlenderBipedEnv, randomization_fn  # noqa: E402

TASKS = {
    # v0 (stand_A8_nodr_s0): quiet-stand shaping only -- learned to slow its fall, never to step
    "stand_quiet": dict(cmd_zero_prob=1.0, rw=dict(both_feet=0.3, touchdown=-0.2)),
    # v1: dynamic balance = stepping in place on the gait clock, zero velocity command
    "stand_v1": dict(cmd_zero_prob=1.0, phase_when_still=True,
                  rw=dict(contact_phase=0.5, swing_height=-100.0, feet_air_time=0.5, ang_vel_xy=-0.02,
                          termination=-5.0, alive=1.0)),
    # v2: v1 + hardware-plausibility pressure (sustained torque above 1/3 stall, speed, saturation)
    "stand": dict(cmd_zero_prob=1.0, phase_when_still=True,
                  rw=dict(contact_phase=0.5, swing_height=-100.0, feet_air_time=0.5, ang_vel_xy=-0.02,
                          termination=-5.0, alive=1.0, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                          saturation=-1.0, action_rate=-0.05)),
    # quiet-stand fine-tune (start from a stepping policy via --init): can it stop stepping?
    "stand_still": dict(cmd_zero_prob=1.0, phase_when_still=False,
                        rw=dict(both_feet=0.5, touchdown=-0.5, ang_vel_xy=-0.02, termination=-5.0, alive=1.0,
                                torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0,
                                action_rate=-0.05)),
    # quiet stand v2 (E29+): also penalize every lift-off (micro-steps never count as touchdowns) and reward
    # zero base velocity -- from scratch or as a fine-tune
    "stand_quiet2": dict(cmd_zero_prob=1.0, phase_when_still=False, track_ang_sigma=0.005,  # penalize yaw creep (E24)
                         rw=dict(both_feet=0.5, touchdown=-0.5, liftoff=-1.0, stop=1.0, ang_vel_xy=-0.05, track_ang=1.0,
                                 termination=-5.0, alive=1.0, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                                 saturation=-1.0, action_rate=-0.05)),
    # stride walking (E3x): slower clock (2.4 steps/s), longer air time, reward forward tip travel per step
    # (up to 6 cm), wider speed range; v5 heading/tracking/hardware terms; stop at zero command (v6)
    "walk_stride": dict(cmd_vx=(0.05, 0.3), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.25,
                        phase_when_still=False, track_sigma=0.02, heading_cmd=True, track_ang_sigma=0.25,
                        gait_hz=1.2, air_time_target=0.25, swing_height=0.03,
                        # slip -30 (default -4): the footed control (E28) "walked" by skating its soles
                        rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, stride=2.0, feet_air_time=2.0, slip=-30.0,
                                contact_phase=1.0, swing_height=-100.0, ang_vel_xy=-0.02, pose=-0.2,
                                termination=-5.0, alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                                saturation=-1.0, action_rate=-0.05)),
    # gentler stride curriculum (E34): fine-tune v6b toward longer steps instead of training strides from scratch
    "walk_stride_ft": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.25,
                           phase_when_still=True, track_sigma=0.02, heading_cmd=True, track_ang_sigma=0.25,
                           gait_hz=1.4, air_time_target=0.2, swing_height=0.03, stride_target=0.05,
                           rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, stride=1.5, feet_air_time=1.0,
                                   slip=-10.0, contact_phase=0.5, swing_height=-100.0, ang_vel_xy=-0.02, pose=-0.2,
                                   termination=-5.0, alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                                   saturation=-1.0, action_rate=-0.05)),
    # E37: step length = speed / cadence. Enforce the clock hard (contact pattern 3.0 at 2.4 steps/s) and command
    # 0.15-0.3 m/s, so the only way to track speed is a 6-12 cm step. Fine-tune from stride_A8_ft.
    "walk_cadence": dict(cmd_vx=(0.15, 0.3), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.15,
                         phase_when_still=True, track_sigma=0.02, heading_cmd=True, track_ang_sigma=0.25,
                         gait_hz=1.2, air_time_target=0.25, swing_height=0.03, stride_target=0.08,
                         rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, stride=1.5, feet_air_time=1.5,
                                 slip=-10.0, contact_phase=3.0, swing_height=-100.0, ang_vel_xy=-0.02, pose=-0.2,
                                 termination=-5.0, alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                                 saturation=-1.0, action_rate=-0.05)),
    # round 4: imitation of a kinematic reference gait (slender_biped/ref_gait.py) to replace the skitter (5-7 steps/s,
    # 1-3 cm) with real steps. ref_q/ref_tip/ref_contact shape the gait; tracking, stop and hardware terms as before.
    "walk_ref": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                     phase_when_still=False, track_sigma=0.02, heading_cmd=True, track_ang_sigma=0.25,
                     gait_hz=1.5, ref_table="ref30",
                     rw=dict(track_lin=2.0, track_ang=1.0, heading=1.0, stop=2.0, ref_q=1.5, ref_tip=1.5,
                             ref_contact=1.0, slip=-10.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                             torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref2: walk_ref marched in place (vx ~0 at a 0.15 command): with track_sigma 0.02 standing still still
    # earned 32 % of the tracking reward while the imitation terms paid in full. Sharper, heavier tracking.
    "walk_ref2": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.25,
                      gait_hz=1.5, ref_table="ref30",
                      rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, slip=-10.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                              torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref3: walk_ref2 (residual) marched in place at 3.4 steps/s, 22 mm clearance, vx ~0: with ref_sigma_q 0.1 the
    # policy could cancel the stance sweep for ~30 % of ref_q. Tight joint kernel: in the reference's kinematics a
    # non-slipping stance foot that sweeps back carries the pelvis forward. slip -30: at -10 a 0.15 m/s stance slide cost
    # only ~0.2 per step, so the marching policy let the stance foot slide 14-18 % of its travel instead of walking.
    "walk_ref3": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.25,
                      gait_hz=1.5, ref_table="ref30", ref_sigma_q=0.02,
                      rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, ref_q=2.0, ref_tip=1.0,
                              ref_contact=0.5, slip=-30.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                              torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref4: walk_ref3 from scratch never survived (52-step episodes at 10 M) and as a fine-tune of the marcher it
    # still marched. Back to walk_ref2's learnable imitation kernel, keep slip -30, and an easier speed range first.
    "walk_ref4": dict(cmd_vx=(0.05, 0.15), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.25,
                      gait_hz=1.5, ref_table="ref30",
                      rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, slip=-30.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                              torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref5: the walk_ref2 "marcher" actually walks (0.08-0.09 m/s in the body frame, MJX and CPU agree) but turns
    # steadily (yaw-rate command pinned at +-0.3 by the heading loop), so net progress along a heading is ~0.
    # walk_ref2 + strong heading hold: heading 3, yaw-rate tracking 2 with a sharper kernel.
    "walk_ref5": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                      gait_hz=1.5, ref_table="ref30",
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, slip=-10.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                              torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref6: walk_ref5 still spun ~160 deg while starting to walk, then ignored the saturated turn command: 80 % single
    # support on a small pad leaves the hip-yaw joints as the only steering, and ref_q pulled them to 0. Free them.
    "walk_ref6": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                      gait_hz=1.5, ref_table="ref30", ref_yaw_weight=0.0,
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, slip=-10.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                              torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref7 (polish, fine-tune only): the d65 walker (99 % reach 2 m, 5.5 cm foot travel) kept stepping at zero command
    # (start/stop 51 %) and ran the hips at ~1.0x rated RMS with 7500 overspeed events. Quiet-stance terms from
    # stand_quiet2 at zero command and twice the torque pressure.
    "walk_ref7": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.25,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                      gait_hz=1.5, ref_table="d65_30", ref_yaw_weight=0.0,
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=3.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, both_feet=0.5, touchdown=-0.3, liftoff=-0.5, slip=-10.0,
                              ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5, torque=-0.4, over_rated=-6.0,
                              overspeed=-4.0, saturation=-1.0, action_rate=-0.05)),
    # walk_ref8 (style fine-tune): the catwalk reference (45 mm track) only narrowed the stance from 116 to 98 mm; the
    # hip-roll residual pushes the feet back out. Penalize lateral tip separation above width_target (cm^2 units).
    "walk_ref8": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.03, 0.03), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                      gait_hz=1.5, ref_table="d65_cat", ref_yaw_weight=0.0, width_target=0.06,
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, width=-0.05, slip=-10.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0,
                              alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0,
                              action_rate=-0.05)),
    # walk_lat (round 5, fine-tune of r4k_cat2_ft): the catwalk walker crabs ~20 deg off its heading and drifts 1-2 m
    # sideways in 20 s, ignoring sideways commands (trained only on +-0.03 m/s). Wider sideways commands and a reward on
    # the gait-cycle-averaged lateral velocity (the per-step term is dominated by the natural sway). Sensor variants
    # (obs_vel / obs_accel / obs_hist) are set per run with --set.
    "walk_lat": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.1, 0.1), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                     phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                     gait_hz=1.5, ref_table="d65_cat", ref_yaw_weight=0.0, width_target=0.06,
                     rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                             ref_contact=0.5, width=-0.05, track_vy=2.0, slip=-10.0, ang_vel_xy=-0.02, pose=0.0,
                             termination=-5.0, alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                             saturation=-1.0, action_rate=-0.05)),
    # walk_line (round 5): velocity feedback alone (even the true velocity) left 60 % of walks > 0.2 m off the line at
    # 2 m, and a wide lateral kernel let the gait shrink back to 2 cm shuffles. Position feedback: the policy sees its
    # odometry-integrated offset from the commanded path (obs_line) and is rewarded on the true offset (line); the
    # sideways speed term averages over exactly one gait cycle (box); a stride term protects the step length.
    "walk_line": dict(cmd_vx=(0.05, 0.25), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                      gait_hz=1.5, ref_table="d65_cat", ref_yaw_weight=0.0, width_target=0.06,
                      vy_avg_mode="box", vy_avg_sigma=0.0025, obs_vel="odo", odo_contact="sensed", obs_line=True,
                      cmd_resample_s=(4.0, 8.0),
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.0,
                              ref_contact=0.5, width=-0.05, track_vy=2.0, line=3.0, stride=1.0, slip=-10.0,
                              ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5, torque=-0.2, over_rated=-3.0,
                              overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # 2026-11-08, F_board150w snowboard pads on the rug. walk_b150: real steps (residual on a 2.4 steps/s reference that
    # lifts the board's lowest edge ~28 mm), heading held, stance slip penalized so it cannot ski; the rug, servo-profile
    # and delay randomization go in with --set (scripts/b150_run.sh). The first reference (b150L_w24, pelvis at constant
    # height) tilted the stance board +-5 deg: without ankles a level sole fixes the shin angle. The policy marched in
    # place on it for 40 M steps. b150L_flats keeps the stance sole level and lets the pelvis bob instead (smoothed: the
    # unsmoothed b150L_flat jumps 11 deg at the knee where the IK task switches).
    "walk_b150": dict(cmd_vx=(0.04, 0.12), cmd_vy=(-0.02, 0.02), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.1,
                      phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                      gait_hz=1.2, ref_table="b150L_flats", ref_residual=True, ref_yaw_weight=0.0, tip_condim=4,
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.5,
                              ref_contact=1.0, slip=-10.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0, alive=0.5,
                              torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # step_b150 (2026-11-08): short quick steps with a pause after each. Joint-space step-and-settle reference
    # (b150L_settle: 0.3 s swing, 0.325 s with both boards down and nothing moving, level staggered pause poses), episodes
    # start in its zero-speed pose, the forward command ramps (0.03 m/s^2) so the first steps from rest are short,
    # a quarter of the commands are "stand" (more starts), body rotation during the pause penalized, softer rugs
    "step_b150": dict(cmd_vx=(0.02, 0.06), cmd_vy=(-0.01, 0.01), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.25,
                      cmd_vx_slew=0.03, reset_ref_pose=True, phase_when_still=False, track_sigma=0.002,
                      heading_cmd=True, track_ang_sigma=0.1, gait_hz=0.8, ref_table="b150L_settle",
                      ref_residual=True, ref_yaw_weight=0.0, tip_condim=4, stride_target=0.03,
                      rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=1.5,
                              ref_contact=1.5, stride=6.0, settle=-0.3, slip=-10.0, ang_vel_xy=-0.02, pose=0.0,
                              termination=-20.0, alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0,
                              saturation=-1.0, action_rate=-0.05)),
    # step2_b150 (2026-11-08, after the first robot trials of b150_step3_L: the next step came before the frame had
    # settled and the centre of mass sat too far forward, so the boards' toes stuck in the rug): one step per second
    # (0.3 s swing, 0.7 s pause), pelvis leaned back 15 deg (b150L_settle2: centre of mass ~21 mm ahead of the board
    # centres instead of ~32), and the settle gate: the clock waits at each lift-off until the body turns slower than
    # 0.3 rad/s (at most 0.5 s). Rewards as in b150_step3_L (progress, stride, ref_swing; no contact/tip imitation)
    "step2_b150": dict(cmd_vx=(0.015, 0.04), cmd_vy=(-0.01, 0.01), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.25,
                       cmd_vx_slew=0.02, reset_ref_pose=True, phase_when_still=False, track_sigma=0.002,
                       heading_cmd=True, track_ang_sigma=0.1, gait_hz=0.5, ref_table="b150L_settle2",
                       ref_residual=True, ref_yaw_weight=0.0, tip_condim=4, stride_target=0.03, vy_avg_mode="box",
                       settle_gate_u=(0.17, 0.67), settle_gate_w=0.3, settle_gate_max=0.5,
                       rw=dict(track_lin=1.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=0.0,
                               ref_contact=0.0, ref_swing=3.0, progress=4.0, stride=20.0, settle=-0.3, slip=-10.0,
                               ang_vel_xy=-0.02, pose=0.0, termination=-10.0, alive=0.5, torque=-0.2,
                               over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # shift_b150 (2026-11-09): slow alternating steps on the shift-and-step reference (slender_biped/shift_gait.py,
    # b150L_shift: per 2.5 s step: settle 0.4 s, hip-roll weight shift 0.8 s (stance leg inward, swing leg outward,
    # 0.10 rad), 0.35 s swing with the shift blended out over the swing + 0.4 s, settle). The settle gate holds the
    # clock at each lift-off until the body turns slower than 0.3 rad/s (up to 1 s): rebalance fully between steps.
    "shift_b150": dict(cmd_vx=(0.006, 0.016), cmd_vy=(0.0, 0.0), cmd_wz=(-0.2, 0.2), cmd_zero_prob=0.15,
                       cmd_vx_slew=0.01, reset_ref_pose=True, phase_when_still=False, track_sigma=0.0005,
                       heading_cmd=True, track_ang_sigma=0.1, gait_hz=0.2, ref_table="b150L_shift",
                       ref_residual=True, ref_yaw_weight=0.0, tip_condim=4, stride_target=0.03, vy_avg_mode="ema", vy_avg_tau=5.0,  # a 250-sample box buffer made training ~5x slower
                       settle_gate_u=(0.235, 0.735), settle_gate_w=0.3, settle_gate_max=1.0,
                       rw=dict(track_lin=1.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=0.0,
                               ref_contact=0.0, ref_swing=6.0, progress=4.0, stride=20.0, settle=-0.3, slip=-10.0,
                               ang_vel_xy=-0.05, pose=0.0, termination=-20.0, alive=1.0, torque=-0.2,
                               over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # bal_b150 (2026-11-09): balance on the left board: shift, lift the right board, hold it up 2 s, lower, unshift,
    # settle (b150L_bal1, 6 s cycle). Paid for keeping the right board up during the hold (ref_swing) and upright.
    "bal_b150": dict(cmd_vx=(0.01, 0.01), cmd_vy=(0.0, 0.0), cmd_wz=(0.0, 0.0), cmd_zero_prob=0.0,
                     reset_ref_pose=True, phase_when_still=False, heading_cmd=True, track_ang_sigma=0.1,
                     gait_hz=1.0 / 6.0, ref_table="b150L_bal1", ref_residual=True, ref_yaw_weight=0.0, tip_condim=4,
                     rw=dict(track_lin=0.0, track_ang=1.0, heading=2.0, stop=0.0, ref_q=0.5, ref_tip=0.0,
                             ref_contact=0.0, ref_swing=8.0, progress=0.0, stride=0.0, settle=0.0, slip=-10.0,
                             ang_vel_xy=-0.05, pose=0.0, termination=-20.0, alive=1.0, torque=-0.2,
                             over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # ank2 (2026-11-09): the ankle-roll leg as designed (hardware/ankle_p1 v2: servo in the foot, straight shin, straight
    # rest pose; models/hw_ank2_<pad>_legs_10dof). Reference slender_biped/ankle_gait.py: flat feet throughout (shins
    # vertical, the rear knee bent forward), weight shift by leaning both legs, swing-foot clearance by hip hike, the
    # foot carried at constant height and lowered slowly; 4 s per step, a full rebalance in between. land_v: soft touchdown.
    "ank2_step": dict(cmd_vx=(0.004, 0.008), cmd_vy=(0.0, 0.0), cmd_wz=(-0.1, 0.1), cmd_zero_prob=0.15,
                      cmd_vx_slew=0.004, reset_ref_pose=True, phase_when_still=False, track_sigma=0.0002,
                      heading_cmd=True, track_ang_sigma=0.1, gait_hz=0.125, ref_table="a2_step",
                      ref_residual=True, ref_yaw_weight=0.0, tip_condim=4, stride_target=0.05, vy_avg_mode="ema",
                      vy_avg_tau=8.0, ref_swing_h=0.005, settle_gate_u=(0.175, 0.675), settle_gate_w=0.3,
                      settle_gate_max=1.0,
                      rw=dict(track_lin=1.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=0.0,
                              ref_contact=0.0, ref_swing=6.0, progress=4.0, stride=20.0, settle=-0.3, slip=-10.0,
                              land_v=-200.0, ang_vel_xy=-0.1, pose=0.0, termination=-20.0, alive=1.0, torque=-0.2,
                              over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    "ank2_bal": dict(cmd_vx=(0.01, 0.01), cmd_vy=(0.0, 0.0), cmd_wz=(0.0, 0.0), cmd_zero_prob=0.0,
                     reset_ref_pose=True, phase_when_still=False, heading_cmd=True, track_ang_sigma=0.1,
                     gait_hz=1.0 / 5.8, ref_table="a2_bal", ref_residual=True, ref_yaw_weight=0.0, tip_condim=4,
                     ref_swing_h=0.005,
                     rw=dict(track_lin=0.0, track_ang=1.0, heading=2.0, stop=0.0, ref_q=0.5, ref_tip=0.0,
                             ref_contact=0.0, ref_swing=8.0, progress=0.0, stride=0.0, settle=0.0, slip=-10.0,
                             land_v=-200.0, ang_vel_xy=-0.1, pose=0.0, termination=-20.0, alive=1.0, torque=-0.2,
                             over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # ski_b150: cross-country "classic" stride: the boards slide alternately and never leave the floor (airborne
    # penalized, no slip penalty), on a 2 strides/s level-sole reference whose gliding board barely skims (8 mm lift
    # at the centre, the edge stays on the floor)
    "ski_b150": dict(cmd_vx=(0.04, 0.12), cmd_vy=(-0.02, 0.02), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.1,
                     phase_when_still=False, track_sigma=0.005, heading_cmd=True, track_ang_sigma=0.1,
                     gait_hz=1.0, ref_table="b150L_skiflat", ref_residual=True, ref_yaw_weight=0.0, tip_condim=4,
                     airborne_h=0.006,   # the gliding board's edge drags; only a clearly lifted board counts (1.5 mm
                                         # with weight -1 cancelled the reference's glide: the skier stood still)
                     rw=dict(track_lin=3.0, track_ang=2.0, heading=3.0, stop=2.0, ref_q=1.0, ref_tip=0.0,
                             ref_contact=0.0, airborne=-0.5, slip=0.0, ang_vel_xy=-0.02, pose=0.0, termination=-5.0,
                             alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0,
                             action_rate=-0.05)),
    "walk_v1": dict(cmd_vx=(0.0, 0.2), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                 phase_when_still=True,
                 rw=dict(track_lin=1.5, feet_air_time=1.0, contact_phase=0.5, swing_height=-100.0,
                         ang_vel_xy=-0.02, pose=-0.2, termination=-5.0, alive=1.0)),
    # v3: v2 never tracked velocity (stepped in place, drifted backward); stronger/wider tracking, less alive
    # v6 (fine-tune from v5): v5 kept walking at ~0.09 m/s on zero command -> sharp stop reward, more zeros
    "walk_v6": dict(cmd_vx=(0.0, 0.2), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.35,
                    phase_when_still=True, track_sigma=0.02, heading_cmd=True, track_ang_sigma=0.25,
                    rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, stop=2.0, feet_air_time=1.0,
                            contact_phase=0.5, swing_height=-100.0, ang_vel_xy=-0.02, pose=-0.2, termination=-5.0,
                            alive=0.5, torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0,
                            action_rate=-0.05)),
    # v5: v4 + wide yaw-rate kernel (0.25) + direct heading-hold reward -- v4 yaw reward stayed ~0.06 of max
    "walk": dict(cmd_vx=(0.0, 0.2), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                 phase_when_still=True, track_sigma=0.02, heading_cmd=True, track_ang_sigma=0.25,
                 rw=dict(track_lin=3.0, track_ang=1.0, heading=1.0, feet_air_time=1.0, contact_phase=0.5,
                         swing_height=-100.0, ang_vel_xy=-0.02, pose=-0.2, termination=-5.0, alive=0.5,
                         torque=-0.2, over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    # v4: v3 + heading command (corrective yaw rate) and stronger yaw tracking -- v3 drifted ~80 deg in 20 s
    "walk_v4": dict(cmd_vx=(0.0, 0.2), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                 phase_when_still=True, track_sigma=0.02, heading_cmd=True,
                 rw=dict(track_lin=3.0, track_ang=1.5, feet_air_time=1.0, contact_phase=0.5, swing_height=-100.0,
                         ang_vel_xy=-0.02, pose=-0.2, termination=-5.0, alive=0.5, torque=-0.2,
                         over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    "walk_v3": dict(cmd_vx=(0.0, 0.2), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                 phase_when_still=True, track_sigma=0.02,
                 rw=dict(track_lin=3.0, track_ang=0.5, feet_air_time=1.0, contact_phase=0.5, swing_height=-100.0,
                         ang_vel_xy=-0.02, pose=-0.2, termination=-5.0, alive=0.5, torque=-0.2,
                         over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
    "walk_v2": dict(cmd_vx=(0.0, 0.2), cmd_vy=(-0.05, 0.05), cmd_wz=(-0.3, 0.3), cmd_zero_prob=0.2,
                 phase_when_still=True,
                 rw=dict(track_lin=1.5, feet_air_time=1.0, contact_phase=0.5, swing_height=-100.0,
                         ang_vel_xy=-0.02, pose=-0.2, termination=-5.0, alive=1.0, torque=-0.2,
                         over_rated=-3.0, overspeed=-2.0, saturation=-1.0, action_rate=-0.05)),
}


def widen(restore, env):
    """Insert zero-weight input rows in FRONT of an older checkpoint's first layers for the round-5 extra inputs,
    with the normalizer for the new inputs seeded from env defaults (the running count is huge, so they would
    otherwise stay at mean 0 / std 1)."""
    norm, pol, val = restore
    mean, std, sv = dict(norm.mean), dict(norm.std), dict(norm.summed_variance)
    count = float(np.asarray(sv["state"])[0] / np.asarray(std["state"])[0] ** 2)
    for key, net, size, stats in (("state", pol, env.observation_size["state"][0],
                                   lambda: env.extra_obs_stats(np.asarray(mean["state"]), np.asarray(std["state"]))),
                                  ("privileged_state", val, env.observation_size["privileged_state"][0],
                                   env.priv_extra_stats)):
        k = net["params"]["hidden_0"]["kernel"]
        n = size - k.shape[0]
        if n == 0:
            continue
        m0, s0 = stats()
        assert len(m0) == n, (key, n, len(m0))
        net["params"]["hidden_0"]["kernel"] = np.concatenate([np.zeros((n, k.shape[1]), k.dtype), np.asarray(k)], 0)
        dt = np.asarray(mean[key]).dtype
        mean[key] = np.concatenate([m0.astype(dt), np.asarray(mean[key])])
        std[key] = np.concatenate([s0.astype(dt), np.asarray(std[key])])
        sv[key] = np.concatenate([(count * s0 ** 2).astype(dt), np.asarray(sv[key])])
        print(f"[widen] {key}: +{n} inputs", flush=True)
    return (norm.replace(mean=mean, std=std, summed_variance=sv), pol, val)


def build_cfg(a) -> EnvConfig:
    base = EnvConfig()
    over = dict(TASKS[a.task])
    rw = dict(base.rw)
    rw.update(over.pop("rw", {}))
    for kv in a.rw or []:
        k, v = kv.split("=")
        rw[k] = float(v)
    over.update(model=a.model, task=a.task, dr=a.dr, push_vel=a.push, rw=rw, strength_scale=a.strength,
                obs_noise=a.obs_noise, action_scale=a.action_scale)
    for kv in a.set or []:
        k, v = kv.split("=")
        cur = getattr(base, k)
        over[k] = type(cur)(json.loads(v)) if not isinstance(cur, tuple) else tuple(json.loads(v))
    return dataclasses.replace(base, **over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="baseline_A_8dof")
    ap.add_argument("--task", default="stand", choices=list(TASKS))
    ap.add_argument("--steps", type=float, default=100e6)
    ap.add_argument("--dr", action="store_true")
    ap.add_argument("--push", type=float, default=0.0)
    ap.add_argument("--strength", type=float, default=1.0)
    ap.add_argument("--obs-noise", type=float, default=1.0)
    ap.add_argument("--action-scale", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-envs", type=int, default=4096)
    ap.add_argument("--episode-length", type=int, default=1000)
    ap.add_argument("--init", help="params.pkl to warm-start from (curriculum)")
    ap.add_argument("--rw", nargs="*", help="reward weight overrides k=v")
    ap.add_argument("--set", nargs="*", help="EnvConfig overrides k=json")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--entropy", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.99)
    a = ap.parse_args()

    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg(a)
    env = SlenderBipedEnv(cfg)
    (out / "config.json").write_text(json.dumps({"args": vars(a), "env": dataclasses.asdict(cfg),
                                                 "jax_devices": [str(d) for d in jax.devices()]}, indent=1))
    prog = open(out / "progress.jsonl", "a")
    t0 = time.time()

    def progress(step, metrics):
        rec = {"t": round(time.time() - t0, 1), "step": int(step)}
        for k, v in metrics.items():
            if k.startswith("eval/") and ("episode_" in k or "avg_episode_length" in k) or k.startswith("training/"):
                try:
                    rec[k] = float(np.asarray(v))
                except Exception:
                    pass
        prog.write(json.dumps(rec) + "\n")
        prog.flush()
        print(f"[{rec['t']:7.0f}s] step {step/1e6:7.1f}M  reward {rec.get('eval/episode_reward', float('nan')):8.2f}  "
              f"len {rec.get('eval/avg_episode_length', float('nan')):6.1f}", flush=True)

    def save_params(step, make_policy, params):
        with open(out / f"params_{int(step/1e6):04d}M.pkl", "wb") as f:
            pickle.dump(jax.device_get(params), f)
        with open(out / "params.pkl", "wb") as f:
            pickle.dump(jax.device_get(params), f)

    restore = None
    if a.init:
        with open(a.init, "rb") as f:
            restore = widen(pickle.load(f), env)

    net = functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128, 128, 128),
                            value_hidden_layer_sizes=(256, 256, 256, 256, 256),
                            policy_obs_key="state", value_obs_key="privileged_state")
    try:
        make_inference, params, _ = ppo.train(
            environment=env, num_timesteps=int(a.steps), num_envs=a.num_envs, episode_length=a.episode_length,
            randomization_fn=randomization_fn if a.dr else None,
            learning_rate=a.lr, entropy_cost=a.entropy, discounting=a.gamma, unroll_length=20, batch_size=256,
            num_minibatches=32, num_updates_per_batch=4, normalize_observations=True, reward_scaling=1.0,
            clipping_epsilon=0.2, gae_lambda=0.95, max_grad_norm=1.0, network_factory=net, seed=a.seed,
            num_evals=max(2, int(a.steps // 10e6) + 1), num_eval_envs=512, deterministic_eval=True, progress_fn=progress,
            # restoring the critic too: E16 (v6) collapsed when fine-tuning with a freshly initialized critic
            policy_params_fn=save_params, restore_params=restore, restore_value_fn=restore is not None,
            action_repeat=1)
        save_params(a.steps, None, params)
        (out / "done.json").write_text(json.dumps({"ok": True, "wall_s": time.time() - t0}))
    except Exception:
        (out / "done.json").write_text(json.dumps({"ok": False, "error": traceback.format_exc()}))
        raise


if __name__ == "__main__":
    main()
