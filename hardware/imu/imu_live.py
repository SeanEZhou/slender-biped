# Live IMU readout for an MPU6050/MPU6500 on an RP2040 running MicroPython. Runs ON the board:
#     mpremote connect /dev/ttyACM0 run hardware/imu/imu_live.py        (Ctrl-C to stop)
#
# Wiring as tested 2026-11-05: I2C0, SDA = GP0, SCL = GP1, address 0x68. The chip answers WHO_AM_I = 0x70,
# i.e. an MPU6500 sold as an "MPU6050"; the accel/gyro registers are the same.
# Hardware I2C failed every transfer on this wiring (weak pull-ups / long leads), so this uses SoftI2C,
# which was 100 % reliable. Adding 2.2-4.7 kOhm pull-ups to 3V3 on SDA/SCL should let hardware I2C work.
#
# Keep the board still for the first 2 s: it measures the gyro bias. Then tilt and turn it:
# roll/pitch come from a complementary filter (gyro integration, corrected toward the accelerometer).
from machine import SoftI2C, Pin
import math
import struct
import time

SDA, SCL, ADDR = 0, 1, 0x68
ACC_LSB, GYR_LSB = 16384.0, 131.0            # +-2 g, +-250 dps (the defaults set below)
ALPHA = 0.98                                 # complementary filter: weight on the gyro
DT_MS = 20                                   # 50 Hz

i2c = SoftI2C(sda=Pin(SDA), scl=Pin(SCL), freq=100000)
who = i2c.readfrom_mem(ADDR, 0x75, 1)[0]
print("WHO_AM_I = 0x%02x (%s)" % (who, {0x68: "MPU6050", 0x70: "MPU6500", 0x71: "MPU9250"}.get(who, "unknown")))
i2c.writeto_mem(ADDR, 0x6B, b"\x00")         # wake
time.sleep_ms(100)
i2c.writeto_mem(ADDR, 0x1C, b"\x00")         # accel +-2 g
i2c.writeto_mem(ADDR, 0x1B, b"\x00")         # gyro +-250 dps
i2c.writeto_mem(ADDR, 0x1A, b"\x03")         # DLPF ~44 Hz: less noise at 50 Hz sampling


def read():
    ax, ay, az, _t, gx, gy, gz = struct.unpack(">hhhhhhh", i2c.readfrom_mem(ADDR, 0x3B, 14))
    return (ax / ACC_LSB, ay / ACC_LSB, az / ACC_LSB), (gx / GYR_LSB, gy / GYR_LSB, gz / GYR_LSB)


print("hold still: measuring gyro bias...")
n, bias = 100, [0.0, 0.0, 0.0]
for _ in range(n):
    _, g = read()
    bias = [b + v / n for b, v in zip(bias, g)]
    time.sleep_ms(DT_MS)
print("gyro bias (dps): %+.2f %+.2f %+.2f" % tuple(bias))

(ax, ay, az), _ = read()
roll = math.degrees(math.atan2(ay, az))
pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
yaw = 0.0
last = time.ticks_ms()
print("   ax     ay     az   |a|  |     gx      gy      gz dps |   roll  pitch    yaw")
k = 0
while True:
    (ax, ay, az), g = read()
    gx, gy, gz = (v - b for v, b in zip(g, bias))
    now = time.ticks_ms()
    dt = time.ticks_diff(now, last) / 1000
    last = now
    acc_roll = math.degrees(math.atan2(ay, az))
    acc_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
    roll = ALPHA * (roll + gx * dt) + (1 - ALPHA) * acc_roll
    pitch = ALPHA * (pitch + gy * dt) + (1 - ALPHA) * acc_pitch
    yaw += gz * dt                           # gyro only: drifts slowly, no magnetometer
    k += 1
    if k % 5 == 0:                           # print at 10 Hz
        print("\r%+.2f  %+.2f  %+.2f  %.2f | %+7.1f %+7.1f %+7.1f    | %+6.1f %+6.1f %+6.1f  "
              % (ax, ay, az, math.sqrt(ax * ax + ay * ay + az * az), gx, gy, gz, roll, pitch, yaw), end="")
    time.sleep_ms(max(0, DT_MS - time.ticks_diff(time.ticks_ms(), now)))
