# Torso P0

The T1 torso as printable parts for a Bambu A1 (256 mm cube). Four tubes stack above the P08 plate, and two
removable panels form the triangular hip below it. The plate sticks out of the side openings.
`cad/torso.py` builds everything and writes `torso_report.json`. It needs the posed leg meshes from `cad/export_legs.py` once.

`T1_shape.json` is the one saved copy of the approved shape, before arm openings and slicing. To recut the arm openings
or move the seams, edit the constants at the top of `cad/torso.py` and rerun it. No mesh of the shape is stored.

## Parts

| part | what it is | print orientation | supports |
|---|---|---|---|
| T01 hip_front | front panel, ear under the plate, tongue at the crotch | top face down | tree, under the ear and crotch floor |
| T02 hip_rear | rear panel, ear under the plate | top face down | tree, under the ear and crotch floor |
| T03 pelvis_ring | sits on the plate, has the bolting floor | floor down | none expected |
| T04 midriff | waist to under the chest | upright | none expected; use a brim |
| T05 chest | lower half of the arm openings | upright | check the slicer preview; use a brim |
| T06 shoulders | upper half of the arm openings, neck | upright | tree, inside the shoulder slope and arm arches; use a brim |

Files in `exports/print_ready/` are already oriented and centred on the bed. Walls are 2.4 mm: use 0.4 mm nozzle,
0.2 mm layers, and enough perimeters to fill the wall. Hide the seam toward the back.

## Thread test: print this first

`fit_tests/m3_thread_test_print.stl` is one small flat plate with through holes from 2.2 to 2.9 mm, each engraved with its size.
Drive one of the spare M3x6 servo screws into each hole. Set `M3_PILOT` in `cad/torso.py` to the size that drives firmly and holds without
stripping or splitting, then rerun `cad/torso.py`. The design value of 2.6 is a placeholder until then.

## Assembly

1. Slide T01 in from the front and T02 from the rear. The ears go under the plate and the tongue over T02's floor.
2. Fit an M3x6 screw up through T02's floor into the tongue.
3. Drop M3 nuts into the ear pockets. Set T03 on the plate.
4. Fit six M3x12 down through T03's floor and the plate: four into the ear nuts, two (x = 0) into loose nuts under the plate.
5. Stack T04, T05 and T06, each over the sleeve of the piece below, and fit the M3x6 screws from outside.

To reach the plate later, remove the M3x6 screws and lift off T06, T05 and T04. The bolting floor is then in reach
through T03's top opening. Removing the hip panels also takes out the four ear bolts.

## Not verified

- Nothing has been printed. The thread test only sets the pilot size. The tube seams' pilots print sideways, so they may
  come out slightly different. The 0.15 mm per-side sleeve fit and the panel clearances are also untested, so print
  T03 and T04 next and test that seam before printing the rest.
- The clearance checks use the CAD legs. Real-robot joint limits and cabling aren't covered, and no cable routes are modelled.
- No arm or head mounts exist yet: the arm openings and neck are plain holes.
- The walking policies were trained without this mass. Add it to the sim and retrain before walking with the torso on.
