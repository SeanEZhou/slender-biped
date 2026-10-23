# Actuator research

The specs live in `configs/actuators.yaml`, and every entry lists its sources. Torques are in
N·m and speeds in rad/s. Prices were observed in October 2026 and are planning numbers, not quotes.
**No purchases have been made in this phase.**

| model | V | stall N·m | rated N·m | no-load rad/s | mass g | case L×W×H mm | ~USD | spec source |
|---|---|---|---|---|---|---|---|---|
| Feetech STS3215-C046 (1:147) | 7.4 | 1.41 | 0.47 (vendor) | 11.1 | 55 | 45.2×24.7×35 | 25 | retailer listings (no official datasheet found) |
| Feetech STS3215-C018 (1:345) | 12 | 2.94 | 0.98 (vendor) | 4.7 | 55 | 45.2×24.7×35 | 32 | retailer listings |
| Feetech STS3032 | 6 | 0.44 | 0.15 (vendor) | 11.6 | 20 | 23.2×12.1×28.5 | 15 | retailer listings |
| ROBOTIS XL330-M288-T | 5 | 0.52 | **unknown** | 10.8 | 18 | 34×20×26 | 24 | official e-manual |
| ROBOTIS XC330-T288-T | 11.1 | 0.92 | **unknown** | 6.8 | 23 | 34×20×26 | 103 | official e-manual |
| ROBOTIS XL430-W250-T | 11.1 | 1.4 | **unknown** | 6.0 | 57 | 46.5×28.5×34 | ~50 (unverified) | official e-manual |
| Waveshare SC15 | 8.4 | 1.67 ("17 kg·cm") | unknown | 6.5 (6 V figure) | ~60 (not listed) | 40.5×40×20 | ~20 (unverified) | Waveshare wiki |

Notes
- Every "kg·cm" figure is a stall-equivalent. 1 kgf·cm = 0.0981 N·m.
- ROBOTIS documents no continuous rating, and its manual warns that stall torque is not the
  continuous output. Feetech vendor "rated" figures come from retailer listings at about 1/3 of
  stall, and we have not verified them.
- SC15: the wiki does not list mass, gear ratio or an 8.4 V speed. At 40 mm wide it is also
  too bulky for the leg diameter, so it was dropped early.
- XL430: torque is similar to the STS3215-C046, but the case is bigger and the speed lower.
  It offers no advantage here.

Sources
- https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
- https://emanual.robotis.com/docs/en/dxl/x/xc330-t288/
- https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/
- https://en.robotis.com/shop_en/category.php?ca_id=20
- https://www.robotshop.com/products/robotis-dynamixel-xc330-t288-t-smart-servo-actuator
- https://www.waveshare.com/wiki/SC15_Servo
- https://www.robotshop.com/products/feetech-12v-30kgcm-magnetic-encoding-servo-sts3215
- https://openelab.io/products/feetech-sts3215-c018-servo-12v
- https://evelta.com/sts3215-c046-7-4v-1-147-magnetic-encoding-metal-gear-dual-axis-ttl-serial-bus-servo-motor/
- https://eckstein-shop.de/Motors?mf=1580 (STS3032)

## Assignment sets

| set | yaw | roll / pitch / knee | ankle | servo cost (10 servos) | voltage rails |
|---|---|---|---|---|---|
| **baseline** | STS3032 | STS3215-C046 | STS3032 | ≈ $210 | 2S 7.4 V + 6 V buck |
| torque_feetech_12v | STS3032 | STS3215-C018 | STS3032 | ≈ $250 | 12 V + 6 V buck |
| slim_dynamixel | XL330 | XC330-T288 | XL330 | ≈ $715 | 11.1 V + 5 V buck |
| xl330_only | XL330 ×10 | | | ≈ $240 | 5 V |

All Feetech servos share one SCS/STS TTL bus and protocol. The two Dynamixel sets use
Protocol 2.0.
