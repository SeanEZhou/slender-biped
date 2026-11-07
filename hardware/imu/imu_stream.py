# IMU streaming firmware for the RP2040 + MPU6050/6500 (MicroPython). Install it as the board's main.py so it
# streams on every power-up:
#     mpremote connect /dev/ttyACM0 cp hardware/imu/imu_stream.py :main.py + reset
#
# Output, 100 lines/s on the USB serial port, in the SENSOR's own axes (the host maps them to the pelvis):
#     I <gx> <gy> <gz> <wx> <wy> <wz> <seq>
#   g = gravity direction, unit vector pointing DOWN (complementary filter: gyro-propagated, pulled toward the
#       accelerometer with a ~0.25 s time constant), w = angular rate in rad/s with the startup bias removed,
#   seq = sample counter (the host checks for gaps).
# Input (single characters on the same port): 'b' = re-measure the gyro bias (hold still for 1 s).
#
# Wiring as tested: I2C0, SDA = GP0, SCL = GP1, address 0x68. SoftI2C, because hardware I2C failed on this
# wiring (weak pull-ups; see README section 13).
import math
import select
import struct
import sys
import time

from machine import Pin, SoftI2C

SDA, SCL, ADDR = 0, 1, 0x68
ACC_LSB = 16384.0                      # +-2 g
GYR_RAD = math.pi / 180 / 131.0        # +-250 dps -> rad/s per LSB
DT_US = 5000                           # 200 Hz filter rate
PRINT_EVERY = 2                        # -> 100 Hz output
ALPHA = 0.02                           # accelerometer weight per 5 ms step (time constant ~0.25 s)

i2c = SoftI2C(sda=Pin(SDA), scl=Pin(SCL), freq=100000)    # 100 kHz: tested reliable; 14 bytes take ~1.8 ms


def setup():
    i2c.writeto_mem(ADDR, 0x6B, b"\x00")       # wake
    time.sleep_ms(100)
    i2c.writeto_mem(ADDR, 0x1C, b"\x00")       # accel +-2 g
    i2c.writeto_mem(ADDR, 0x1B, b"\x00")       # gyro +-250 dps
    i2c.writeto_mem(ADDR, 0x1A, b"\x03")       # DLPF ~44 Hz


def read():
    ax, ay, az, _t, gx, gy, gz = struct.unpack(">hhhhhhh", i2c.readfrom_mem(ADDR, 0x3B, 14))
    return ax / ACC_LSB, ay / ACC_LSB, az / ACC_LSB, gx * GYR_RAD, gy * GYR_RAD, gz * GYR_RAD


def read_retry():
    """read(), but wait out a missing sensor (loose wire) with a message instead of crashing."""
    t_msg = 0
    while True:
        try:
            return read()
        except OSError:
            if time.ticks_diff(time.ticks_ms(), t_msg) > 1000:
                sys.stdout.write("E no answer from the IMU at 0x%02x on SDA=GP%d SCL=GP%d: check the wiring\n"
                                 % (ADDR, SDA, SCL))
                t_msg = time.ticks_ms()
            time.sleep_ms(50)
            try:
                setup()
            except OSError:
                pass


def measure_bias(n=200):
    s = [0.0, 0.0, 0.0]
    for _ in range(n):
        r = read_retry()
        s[0] += r[3]; s[1] += r[4]; s[2] += r[5]
        time.sleep_us(DT_US)
    return [v / n for v in s]


def norm(x, y, z):
    n = math.sqrt(x * x + y * y + z * z) or 1.0
    return x / n, y / n, z / n


for attempt in range(10):          # the sensor can miss the first transfers right after power-up
    try:
        setup()
        break
    except OSError:
        time.sleep_ms(200)
bias = measure_bias()
ax, ay, az, _, _, _ = read_retry()
g = norm(-ax, -ay, -az)            # at rest the accelerometer reads +1 g UP; gravity points down
poll = select.poll()
poll.register(sys.stdin, select.POLLIN)
seq = 0
k = 0
t_next = time.ticks_us()
while True:
    ax, ay, az, wx, wy, wz = read_retry()
    wx -= bias[0]; wy -= bias[1]; wz -= bias[2]
    dt = DT_US * 1e-6
    # a world-fixed vector seen from the rotating body: dg/dt = -w x g
    gx = g[0] - (wy * g[2] - wz * g[1]) * dt
    gy = g[1] - (wz * g[0] - wx * g[2]) * dt
    gz = g[2] - (wx * g[1] - wy * g[0]) * dt
    a = norm(-ax, -ay, -az)
    g = norm((1 - ALPHA) * gx + ALPHA * a[0], (1 - ALPHA) * gy + ALPHA * a[1], (1 - ALPHA) * gz + ALPHA * a[2])
    k += 1
    if k % PRINT_EVERY == 0:
        seq += 1
        sys.stdout.write("I %.4f %.4f %.4f %.4f %.4f %.4f %d\n" % (g[0], g[1], g[2], wx, wy, wz, seq))
    if poll.poll(0):
        c = sys.stdin.read(1)
        if c == "b":
            bias = measure_bias()
            sys.stdout.write("B %.4f %.4f %.4f\n" % tuple(bias))
    t_next = time.ticks_add(t_next, DT_US)
    wait = time.ticks_diff(t_next, time.ticks_us())
    if wait > 0:
        time.sleep_us(wait)
    else:
        t_next = time.ticks_us()           # overran: don't try to catch up
