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

## Thread test

`fit_tests/m3_thread_test_print.stl` is one small flat plate with through holes from 2.2 to 2.9 mm, each engraved with its size.
Drive one of the spare M3x6 servo screws into each hole. Set `M3_PILOT` in `cad/torso.py` to the size that drives firmly and holds without
stripping or splitting, then rerun `cad/torso.py`. On the Bambu A1, 2.9 mm drove in and held firm, and the screw didn't pass through
without threading, so `M3_PILOT` is 2.9. Repeat the test on a different printer or filament.

## Assembly

1. With the hip panels off, set T03 on the plate. Fit the two middle M3x12 (x = 0) down through T03's floor and the plate into loose nuts underneath.
2. Press M3 nuts into the ear pockets and hold them with a strip of tape, since the pockets face down once fitted.
3. Slide T01 in from the front and T02 from the rear. The ears go under the plate and the tongue over T02's floor.
4. Fit an M3x6 screw up through T02's floor into the tongue.
5. Fit the other four M3x12 down through T03's floor and the plate into the ear nuts. They end flush with the nuts; use M3x14 if they don't catch.
6. Stack T04, T05 and T06, each over the sleeve of the piece below, and fit the M3x8 screws from outside. The two front screws at the T05/T06 seam angle downward, square to the sloped chest.

## Head mount

The top 8 mm of the neck works like the other sleeves: fit a head over it and fix it with three M3x6 screws from outside.
The pilots are 2.9 mm, 4 mm below the rim, at the back and 75 degrees either side of the front. The pads behind them
leave an 18.5 mm clear circle down the neck for a future spine, shaft or cables.

For the head's socket, the neck is widest at the bottom of that overlap: 26.4 mm front to back and 26.5 mm side to side, so
bore it to those sizes plus 0.15 mm per side. Use a 2.4 mm wall with 3.4 mm clearance holes, so an M3x6 ends inside the 4 mm pad.

To reach the plate later, remove the seam screws and lift off T06, T05 and T04. The bolting floor is then in reach
through T03's top opening. Removing the hip panels also takes out the four ear bolts.

## Not verified

- No torso part has been printed; only the thread test has. It printed its holes vertically in a 6 mm plate, while the tube
  seams' pilots print sideways or at an angle and grip less depth. Thread depth with a seated head, M3x6 / M3x8: T03/T04 3.0 / 5.0 mm,
  T04/T05 1.8 / 3.8 mm, T05/T06 3.4 / 5.4 mm at the back and 3.9 / 5.9 mm at the front.
  The 0.15 mm per-side sleeve fit and the panel clearances are also untested, so print T03 and T04 next and test that seam
  before printing the rest.
- The clearance checks use the CAD legs. Real-robot joint limits and cabling aren't covered, and no cable routes are modelled.
- No arm mounts exist yet: the arm openings are plain holes. The neck pilots and head socket sizes are also unprinted.
- The walking policies were trained without this mass. Add it to the sim and retrain before walking with the torso on.
