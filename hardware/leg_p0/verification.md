# P0 leg: verification

Re-run all automated checks with `.venv-cad/bin/python hardware/leg_p0/cad/build.py check` (≈ 1 min).
Raw results are in [`verification_report.json`](verification_report.json).

## Automated digital checks (last run)

| check | method | result |
|---|---|---|
| CAD regenerates from source | `cad/build.py` (build123d / OpenCascade) | ✅ 7 printed parts, each one valid solid |
| meshes | every exported STL (left + mirrored right) loaded with trimesh | ✅ all 13 watertight, non-empty |
| interference at neutral and stance | exact solid booleans between every pair of items on different links: printed parts, 4 servo dummies (case + main horn + idler horn), M3 horn-screw and M2 case-screw heads, connector keep-outs (plug + 12 mm cable bend) | ✅ no overlap > 1 mm³ |
| real vendor servo in every pocket | the SO-ARM100 `STS3215_03a.step` (horns included) placed in all four joints, boolean against every printed part | ✅ zero overlap |
| screw stack-ups (vs the SO-101 follower, same servo) | horn: M3×6 through a 3.5 mm arm, ≈2.5 mm into the horn (SO-101: same). case: M2×6 through 2.0 mm of wall under the head + 0.35 mm gap, ≈3.6 mm into the case (SO-101: 2.0 mm wall) | ✅ by construction (`fit.horn_screw`, `fit.case_screw`) |
| range of motion | each joint swept in 5° steps from 0 to its sim target, same boolean check | yaw −20…30 ✅ · roll −14…30 ✅ · pitch −30…85 (target −35…100) ⚠ · knee 0…110 (target 0…120) ⚠ |
| servo fit | pockets are the servo dummy grown by `fit.pocket_clearance` (0.35 mm per side), so each dummy fits its cradle by construction | ✅ |
| fastener access | every screw sits on an outside face, or behind an access hole/counterbore reachable at the documented assembly step | ✅ by design; walk through it in `assembly.md` |
| pad orientation | underside normal of the pad at the stance pose (hip 16°, knee 32°) | ✅ (0, 0, −1): level |
| joint axes | positions and directions | see "axes" in the JSON; yaw ∩ roll and yaw ∩ pitch; roll–pitch offset 34.2 mm |
| mass | solid volume × 1.24 g/cm³ × 0.55 fill + 4 × 55 g servos + 25 g hardware | ≈ 163 g printed, ≈ 408 g on the fixture (rough) |
| knee drive | geometry | ✅ the shin fork bolts directly to the knee servo's horns; no belt, pulley or gear stage exists in the model |

**What these checks do not cover:**
- **Strength:** nothing here checks it.
- **Real geometry:** results are only as good as the servo dimensions below, which are **not yet
  measured on your hardware**.
- **Cables:** the check covers keep-out volumes only. Real cable stiffness and routing must be
  checked by hand (assembly step 6).

## Caliper checklist

Measure these on a real STS3215-C001 with its horns fitted. Put any difference greater than 0.2 mm
into [`params.yaml`](params.yaml), then rebuild.

| # | measure | params.yaml | current value (source) |
|---|---|---|---|
| 1 | case length (output end to far end) | `servo.case_len` | 45.4 (STEP; datasheet 45.23) |
| 2 | case width | `servo.case_wid` | 24.8 (STEP; datasheet 24.73) |
| 3 | case thickness between the two mounting faces (not over the raised label/connector areas) | `servo.case_face_z` × 2 | 31.8 (STEP; datasheet "32") |
| 4 | output axis to the output end of the case | `servo.output_x` = case_len/2 − this | 10.2 → 12.5 from centre (datasheet) |
| 5 | main-horn outer face height above the case centre plane (= half thickness + horn standoff) | `servo.horn_face_z` | 18.7 (STEP) |
| 6 | idler-horn outer face height below the case centre plane | `servo.idler_face_z` | 17.7 (STEP) |
| 7 | horn disc diameter, both horns | `servo.horn_d` | 20.0 (STEP) |
| 8 | horn hole spacing (centre-to-centre of adjacent holes = 2 × offset) | `servo.horn_hole_offset` | 4.95 offset; holes take M3×6 (SO-101 guide) |
| 9 | case mounting holes: positions from the far end and from the centreline, diameter, depth | `servo.mount_holes_top/bot`, `mount_hole_pilot_d` | (−16.5/−20.3/4.2, ±10.25), Ø1.5 (STEP) ⚠ |
| 10 | plug + bent cable protrusion from the idler face; connector position | `servo.connector_keepout`, `connector_x` | 12 mm, x = −1.55 (estimate) ⚠ |

Also confirm:
- whether horns, idler horns and their screws came with your servos (the datasheet lists no
  accessories);
- the M3 horn-screw head diameter (`horn_center_screw_head_d`).

## Coupon loop

Coupons are described in [`assembly.md` step 1](assembly.md#1-print-the-coupons-and-calibrate) and
`cad/coupons.py`. Adjust `fit.pocket_clearance`, `fit.m2_clear_d`, `fit.m3_clear_d`,
`fit.m3_nut_af` and `pad.stem` until each coupon fits, then print the real parts.

## Open risks (flagged)

1. **Horn screws (fixed after the first test print).** The first version used M2 holes on the arms,
   but the horns take M3×6 (as on the SO-101), so the arms now have Ø3.4 holes. That version also
   counterbored P03's bottom arm all the way through, so the screw heads had nothing to bear on; fixed.
2. **Self-tapping into the servo case.** These holes cut threads into the servo's plastic case.
   Don't overtighten. Rebuilding many times may strip them.
3. **Connector plugs.** On the hip-pitch and knee servos the inner arm passes within about 1–3 mm of
   the plug at the range limits. Check by hand with power off.
4. **Weak arms.** 3.5 mm arms on a 55 g knee servo plus the shin are adequate for unloaded motion.
   Expect flex. Don't hang weight on the leg.
5. **Slow servo.** The C001 (1:345) is about half the speed of the C046 the simulation assumed.
   Unloaded tests are fine; the RL policies are not transferable.

## Acceptance criteria (spec section 11)

| criterion | status |
|---|---|
| parametric CAD regenerates from a documented command | ✅ `cad/build.py` |
| separate STLs: pelvis fixture, hip chain, thigh, direct knee, shin and tip; no missing mating pieces | ✅ P01–P07 (the knee cradle is part of P05 thigh; the shin fork is part of P06) |
| four C001 dummies fit their mounts; clearances and fastener/connector access documented | ✅ (clearance 0.35 mm/side, pending coupon A) |
| four independent revolute joints; axes and ROM documented; collision checks pass for the selected safe ROM | ✅ for the `bench_config.yaml` limits |
| knee is direct servo drive, no belt or pulley | ✅ |
| thigh and shin match the slender rounded/tapered silhouette in front and side renders | ✅ see `renders/leg_silhouette_dimensioned.png` and `compare_sim_vs_cad.png`, for your review |
| small flat ankle-less 30 × 18 mm pad, horizontal at stance, no ball foot | ✅ |
| exterior taper, curvature and tube dimensions are parameters | ✅ `thigh.*`, `shin.*`, `pad.*` |
| bench fixture mountable; leg clears the table | ✅ by geometry (desk edge on the inner side, groove on the tongue); confirm by hand at step 6 |
| coupons, assembly guide, annotated renders and hardware BOM exist | ✅ |
| a beginner can install or remove each motor without destructive disassembly | ✅ every servo is held by external screws and slides out of its pocket |
| unverified vendor geometry is flagged for measurement | ✅ caliper checklist above |
| no claims of standing, walking or load-bearing | ✅ |
