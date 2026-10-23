import numpy as np

from slender_biped.actuator import ServoBank, ServoParams, available_torque
from slender_biped.config import load_actuators


def bank(n=1):
    act = load_actuators()
    spec = act["catalog"]["STS3215_C046_7V4"]
    p = ServoParams.from_config([spec] * n, act["servo_model"], 0.001)
    p.latency_steps = 0
    b = ServoBank(p, 0.001)
    b.reset(np.zeros(n))
    return b, spec


def test_never_exceeds_stall():
    b, spec = bank()
    for err in [-3, -0.5, 0.5, 3]:
        b.reset([err]); b.command([err])
        for qd in [-20, -5, 0, 5, 20]:
            tau = b.torque(np.zeros(1), np.full(1, qd))
            assert abs(tau[0]) <= spec["stall_torque"] + 1e-9


def test_torque_speed_curve_and_backemf():
    b, spec = bank()
    b.reset([1.0]); b.command([1.0])
    # full duty forward at half no-load speed -> half stall
    tau = b.torque(np.zeros(1), np.full(1, spec["noload_speed"] / 2))
    assert np.isclose(tau[0], spec["stall_torque"] * 0.5, rtol=0.02)
    # beyond no-load speed no forward torque is available
    tau = b.torque(np.zeros(1), np.full(1, spec["noload_speed"] * 1.2))
    assert tau[0] < 0
    assert b.last["over_speed"][0]


def test_saturation_is_logged_not_hidden():
    b, _ = bank()
    b.reset([2.0]); b.command([2.0])
    b.torque(np.zeros(1), np.zeros(1))
    assert b.last["duty_sat"][0] and b.last["u_req"][0] > 1


def test_available_torque_envelope():
    a = available_torque(1.0, 10.0, np.array([-5, 0, 5, 10, 15]), 1.0)
    np.testing.assert_allclose(a, [1.0, 1.0, 0.5, 0.0, 0.0])
