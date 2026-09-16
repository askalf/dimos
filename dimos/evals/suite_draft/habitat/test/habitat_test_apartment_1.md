# Habitat test — apartment_1.glb

## Executable suite

`dimos/evals/suites/habitat/test/habitat_test_apartment_1.py`: **13 cases**, all tagged `draft-reference`. Includes template questions 1–10 and best-supported visual candidates `mirror_shape` (B, circular), `window_covering` (A, horizontal blinds), and `dining_door_state` (B, open). Candidates without usable answers remain below and are not scored. Each case gets independent context and a fresh seed-0 Habitat environment.

The suite defaults to dataset `default` and project-local `target/habitat/data/versioned_data/habitat_test_scenes/apartment_1.glb`. Set `HABITAT_TEST_SCENE` to the inspected asset path below on this machine; optional `HABITAT_TEST_DATASET_CONFIG` overrides the default dataset. This is a direct GLB scene, not ReplicaCAD's `apt_1` handle. No live eval was run; low-camera tabletop visibility remains a review item.

## Scene reference

- **Dataset:** Habitat Test Scenes; this is not ReplicaCAD `apt_1` or the DimSim apartment.
- **Scene asset:** `/home/ruthwik/Documents/habitat-sim/data/versioned_data/habitat_test_scenes/apartment_1.glb`.
- **Navmesh:** adjacent `apartment_1.navmesh`.
- **GLB SHA-256:** `c0b1314d1b948170e4110d1cf1a001c54ef02788092d49586ad73591b871785e`.
- **Navmesh SHA-256:** `3694bf3909ca9668e0506b07e20d932c8243adb2654dffb4f27f8fa8bd9b3c4b`.
- **Inspection:** read the GLB metadata and rendered 40 views from ten seeded navigable positions using Habitat-Sim 0.3.3. This was an authoring inspection, not an agent eval.
- **Metadata limitation:** one textured mesh, one material, one image, and two nodes (`apartment_1` and `Lamp`). Habitat exposed **zero semantic objects and zero semantic regions**, with a semantic-descriptor load error. Node/mesh counts are not physical object/room counts; `Lamp` is not evidence of a single lamp in the scene.
- **Observed spaces:** a furnished lounge with an L-shaped sofa and wall-mounted TV, a separate dining room, and connecting corridor space. Some views contain black scan boundaries/unmodeled space. Do not label these as additional rooms or as evidence that a household object is absent everywhere.
- **Navmesh metadata:** total navigable area approximately **52.8754 m²**, with **two navigation islands**. Neither value is a ground truth for interior floor area or room count: navmesh area reflects traversability and its build settings.
- **Status:** all questions below await human validation. Ground truths in the first section have direct rendered evidence; the second section contains candidates needing further work.

## Questions from templates

### 1. [Object Location] Which room contains the wall-mounted television?

- **ID:** `habitat_test_apartment_1_tv_location`
- **Options:** A) Dining room; B) Bedroom; C) Living room; D) Bathroom.
- **Answer format:** One letter.
- **Ground truth:** **C — Living room.**
- **Scoring:** `exact("C", answer.strip().upper())`.
- **Evidence:** Views V1 and V3 show the TV opposite the upholstered sectional and coffee table in the lounge.
- **Validation:** Confirm the room label; accept “living room” as the label for this lounge, without implying a complete house inventory.

### 2. [Object Location] Which room contains the round wall mirror?

- **ID:** `habitat_test_apartment_1_round_mirror_location`
- **Options:** A) Dining room; B) Bathroom; C) Bedroom; D) Living room.
- **Answer format:** One letter.
- **Ground truth:** **A — Dining room.**
- **Scoring:** `exact("A", answer.strip().upper())`.
- **Evidence:** Views V4 and V6 show the round mirror above a sideboard beside the dining table and chairs.
- **Validation:** Confirm there is no second matching round mirror creating ambiguity.

### 3. [Object Existence] Is there a television in the living room?

- **ID:** `habitat_test_apartment_1_tv_exists`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** The television is directly visible in V1 and V3.
- **Validation:** Easy positive-presence baseline; overlaps with Q1, so consider retaining only one in the final suite.

### 4. [Object Existence] Is there a potted tree in the living room?

- **ID:** `habitat_test_apartment_1_potted_tree_exists`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V2 shows a tall leafy plant in a floor pot near the framed wall art and sofa area.
- **Validation:** Confirm that “potted tree” is an appropriate visual category; no species identification is intended.

### 5. [Object Counting] How many oversized chess-piece decorations are on the console beneath the television?

- **ID:** `habitat_test_apartment_1_chess_decoration_count`
- **Answer format:** One count.
- **Ground truth:** **2.**
- **Scoring:** `exact(2, first_number(answer))`; exact count, no partial credit.
- **Evidence:** V3 shows two large pale chess-piece decorations on the left side of the console. Nearby decorative objects are not included.
- **Validation:** Confirm the count in a close view and check that a low robot-camera viewpoint can distinguish both decorations.

### 6. [Object Counting] How many tiers does the serving stand on the dining table have?

- **ID:** `habitat_test_apartment_1_serving_stand_tiers`
- **Answer format:** One count.
- **Ground truth:** **2.**
- **Scoring:** `exact(2, first_number(answer))`.
- **Evidence:** V4 shows a lower circular tray and a raised upper circular tray joined by a central support and handle.
- **Validation:** Count trays, not the top handle. The authoring camera was higher than the default robot camera; verify observability before including this case.

### 7. [Spatial Relation] What is directly below the round wall mirror?

- **ID:** `habitat_test_apartment_1_below_round_mirror`
- **Options:** A) Bed; B) Sofa; C) Bathtub; D) Sideboard.
- **Answer format:** One letter.
- **Ground truth:** **D — Sideboard.**
- **Scoring:** `exact("D", answer.strip().upper())`.
- **Evidence:** V4 and V6 show the mirror mounted above a dark low cabinet/sideboard.
- **Validation:** “Sideboard” includes the visually identifiable low storage cabinet; this does not require knowing an asset-template name.

### 8. [Spatial Relation] Is the television mounted on the wall above the console?

- **ID:** `habitat_test_apartment_1_tv_above_console`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V1 and V3 show the television above and separated from the console surface.
- **Validation:** Confirm wall mounting visually; do not confuse its position above the console with standing directly on it.

### 9. [Spatial Relation] Is there a coffee table between the sectional sofa and the television?

- **ID:** `habitat_test_apartment_1_coffee_table_between`
- **Answer format:** `yes` or `no`.
- **Ground truth:** **Yes.**
- **Scoring:** `exact("yes", yes_no(answer))`.
- **Evidence:** V1 and the opposite lounge views show the low table occupying the space between the sectional and TV wall.
- **Validation:** Confirm the ordinary floor-plane meaning of “between.” This is a pair-specific version of the relation questions, rather than a universal claim about every table.

### 10. [Object Counting] How many round wall mirrors are visible above the dining-room sideboard?

- **ID:** `habitat_test_apartment_1_sideboard_mirror_count`
- **Answer format:** One count.
- **Ground truth:** **1.**
- **Scoring:** `exact(1, first_number(answer))`.
- **Evidence:** V4 and V6 show one circular wall mirror above that sideboard.
- **Validation:** Scope is this sideboard, not every reflective surface in the entire scan. Easy baseline; potentially redundant with Q2/Q7.

## Other potential questions

Candidates 18–20 are included with best-effort answers as listed above. Other candidates lack usable reference answers and remain unscored.

### 11. [Object Counting] How many chairs surround the dining table?

- **Answer format:** One count.
- **Proposed scoring:** `exact(reference_count, first_number(answer))`.
- **Ground truth:** **Pending.** Multiple chairs are visible in V4–V6, but the initial views do not establish every chair around the table.
- **Reference work:** Inspect both sides and both ends, matching chairs across views to avoid duplicates. Do not infer the count from the number of mesh nodes.

### 12. [Room Counting] How many furnished rooms are in the scanned environment?

- **Answer format:** One count.
- **Proposed scoring:** Exact count after reference validation.
- **Ground truth:** **Pending; at least two distinct furnished rooms were observed.**
- **Reference work:** Inspect the full geometry, entrances, and reachable space. Define whether corridor space counts, and distinguish unmodeled boundaries from room openings. Two navmesh islands do not establish two rooms.

### 13. [Room Size] What is the approximate floor area of the living room, in square meters?

- **Answer format:** One numerical estimate.
- **Proposed scoring:** `numeric(reference_area, first_number(answer), tolerance=t, band=b)`; select tolerances after measuring the floor boundary.
- **Ground truth:** **Pending.**
- **Reference work:** Trace the living-room interior floor polygon from transformed geometry. Include furniture-occupied floor; do not substitute the total scene navmesh area.

### 14. [Object Dimensions] What is the approximate diagonal length of the dining tabletop, in meters?

- **Answer format:** One numerical estimate.
- **Proposed scoring:** Numerical full-credit tolerance and partial-credit band, selected after measurement.
- **Ground truth:** **Pending.**
- **Reference work:** Locate the tabletop corners in the merged mesh, apply the GLB node transform, and measure the horizontal diagonal. Confirm that a rectangular-footprint approximation is appropriate; the scene has no separate table asset bounds.

### 15. [Passage Clearance] What is the largest circular robot radius that fits through the dining-room doorway in 2D, in meters?

- **Answer format:** One numerical estimate.
- **Proposed scoring:** `numeric(min_clear_width / 2, first_number(answer), tolerance=t, band=b)`.
- **Ground truth:** **Pending.**
- **Reference work:** Identify the doorway jambs and measure the narrowest clear horizontal gap. Agree whether the visible door panel affects the gap. Use mesh measurements rather than a navmesh already eroded for an unknown agent radius.

### 16. [Object Displacement] What is the approximate straight-line distance between the television and the round dining-room mirror, in meters?

- **Answer format:** One numerical estimate.
- **Proposed scoring:** Numerical tolerance-band score.
- **Ground truth:** **Pending.**
- **Reference work:** Identify both object centers in the merged mesh and use a documented horizontal-distance convention. Camera positions in the evidence table are not object positions.

### 17. [Spatial Ordering] What is the order of these objects from nearest to farthest from the dining-room doorway?

- **Options:** A) Dining table; B) Round wall mirror; C) Living-room television.
- **Answer format:** All three letters once, in order; contiguous or comma-separated.
- **Proposed scoring:** `rank_order(reference_order, ranking(answer))`; each correctly ordered pair earns one third.
- **Ground truth:** **Pending.**
- **Reference work:** Fix a doorway reference point and specify straight-line distance before measuring. Do not derive a ranking from apparent object size or thumbnail layout.

### 18. [Visual Attribute] What shape is the wall mirror above the dining-room sideboard?

- **Options:** A) Rectangular; B) Circular; C) Triangular; D) Hexagonal.
- **Answer format:** One letter.
- **Candidate ground truth:** **B — Circular**, directly supported by V4/V6.
- **Proposed scoring:** `exact("B", answer.strip().upper())`.
- **Why separate:** A useful new shape-recognition template beyond the current list, but easy and overlapping with the mirror-location questions. The question deliberately does not call it a “round mirror,” which would reveal the answer.

### 19. [Visual Attribute] What type of window covering is used in the living room?

- **Options:** A) Horizontal blinds; B) Fabric curtains; C) Exterior shutters; D) No covering.
- **Answer format:** One letter.
- **Candidate ground truth:** **A — Horizontal blinds**, visible in the lounge views.
- **Proposed scoring:** `exact("A", answer.strip().upper())`.
- **Why separate:** Additional visual-recognition family; validate the scan detail at the intended sensor resolution.

### 20. [Object State] Is the dining-room door open or closed?

- **Options:** A) Closed; B) Open.
- **Answer format:** One letter.
- **Candidate ground truth:** **B — Open**, based on V5 showing a panel angled away from the opening.
- **Proposed scoring:** `exact("B", answer.strip().upper())`.
- **Reference work:** Confirm which panel belongs to this doorway. This is a static scanned appearance, not an articulated door that Habitat can necessarily open or close.

## Evidence views and reproducibility

All positions below are **Habitat world coordinates**, Y-up, in meters; they are authoring camera-support positions, not object locations or benchmark spawns. Camera height was **1.30 m above the listed position**, resolution **480 × 360**, horizontal FOV **90°**. Yaw is a Habitat rotation around +Y; yaw 0 looks along the camera's default -Z direction. Do not reuse these yaw values as ROS yaw without conversion.

| View | Position | Yaw | Evidence |
|---|---|---:|---|
| V1 | `(5.93873, -1.60025, 0.73570)` | 90° | Lounge overview: sofa, low table, television/console, wall art |
| V2 | `(2.00333, -1.50276, -0.56372)` | 90° | Lounge wall art and potted tree |
| V3 | `(3.47248, -1.60025, 0.90150)` | 180° | Television, console, two oversized chess decorations |
| V4 | `(5.51025, -1.60025, 3.48315)` | 180° | Dining table, chairs, two-tier stand, mirror and sideboard |
| V5 | `(6.18316, -1.60025, 6.77649)` | 0° | Opposite dining-room view, chairs and doorway/panel |
| V6 | `(6.18316, -1.60025, 6.77649)` | 90° | Mirror and sideboard close view |

Local authoring artifacts: `/tmp/opencode/apartment1_views.jpg`, close views `/tmp/opencode/apartment1_p3_2.png`, `apartment1_p6_0.png`, `apartment1_p6_1.png`, and `apartment1_p8_2.png`; inspection script `/tmp/opencode/apartment1_inspect.py`. These are temporary local files, not committed reference assets. The asset hashes and view parameters above allow re-rendering after temporary files disappear.

The default Habitat robot camera in the current DimOS connection is **0.45 m high**, lower than this authoring camera. Tabletop questions especially require a low-camera observability check or an explicitly agreed sensor configuration. Positive visual observations here are not proof that the default robot can see them from every start.

## Scoring and validation notes

- Keep the question and answer format in the agent prompt; keep references and evidence notes author-side.
- Reuse `exact`, `first_number`, `yes_no`, `numeric`, `ranking`, and `rank_order` from `dimos/evals/scorers.py`. Catch parser `ValueError` and return zero for unparseable answers.
- Correct counts, yes/no answers, and single-choice letters score 1; incorrect answers score 0. Numerical and ranking questions receive the partial-credit rules from `benchmarkQA.md`.
- `Spatial Relation` above is a proposed pair-specific specialization of the existing `Universal Relation` family; it avoids unnecessary “every” claims in a scene without an annotated inventory.
- No negative whole-house existence question is proposed yet: incomplete scan boundaries and lack of semantic inventory make absence claims harder to establish.
- No refrigerator, kitchen, bedroom, or bathroom inventory is inferred from the filename “apartment.” Do not transfer answers from the DimSim or ReplicaCAD apartments.
- The executable suite uses available best-effort references; the next user pass can adjust overlapping questions and sensor-observability assumptions. No live eval was run.
