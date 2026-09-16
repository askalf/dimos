# HM3D — NBg5UqG3di3

## Executable suite

`dimos/evals/suites/habitat/hm3d/hm3d_NBg5UqG3di3.py`: **10 cases**, all tagged `draft-reference`. Includes the eight template questions and visual candidates `corridor_panel_color` (C, red) and `floor_pattern` (B, herringbone). These use the best available rendered evidence for the next user review. Candidates without usable reference answers are retained below and not scored.

Override `HM3D_DATASET_CONFIG`; default is the project-local example configuration below, full scene handle `00770-NBg5UqG3di3`. Fresh seed-0 Habitat environment per case. Runtime camera visibility and connected navigation remain unverified; no live eval was run.

## Scene reference

- **Full handle:** `00770-NBg5UqG3di3`.
- **Root:** `~/Documents/dimos/target/habitat/data/versioned_data/hm3d-0.2/hm3d/example/`.
- **Dataset config:** `hm3d_example_basis.scene_dataset_config.json`; flat shading, source up `[0,0,1]`, front `[0,1,0]`, no-lights defaults.
- **Scene:** `00770-NBg5UqG3di3/NBg5UqG3di3.basis.glb`, SHA-256 `86ef9e300ae48e67c13899ecb0c6038b6f1addaf282bf9ef22ee34eb36cd49b1`.
- **Navmesh:** `NBg5UqG3di3.basis.navmesh`, SHA-256 `622bdeba91d1408b3ea3b70d9d1464d4e18bf5cd61f52f66368de35bdf67b515`.
- **Inspected metadata:** glTF2.0, `obj2gltf`, 283 nodes/meshes, 88 materials/images. Nodes are scan fragments rather than named furniture instances. Scene directory has only visual GLB/navmesh, no semantic sidecars or XML. Habitat reports semantic-descriptor failure and zero semantic objects/regions.
- **Navigation:** approximately 299.4313 m² navigable area and two islands. All 16 sampled support points had runtime floor Y≈0.10685 m; this is not proof of one complete floor or an architectural floor-area reference.
- **Evidence fallback:** because object/room metadata is absent, rendered 64 authoring views from 16 seeded navigable positions. Observed ornate, largely unfurnished interiors: white paneled corridors, a red-and-gold corridor, a blue-wall/wood-paneled room with a carved fireplace, and a pale-blue decorative room. Do not assume this scan is a standard furnished apartment.

All questions below are draft for human review. The first section is visually supported; metadata supports scene identity/orientation but not the object labels. No whole-scene absence claim is made from sampled views.

## Questions from templates

ID prefix `hm3d_NBg5UqG3di3_`. View codes refer to the reproducibility table below.

| Suffix / template | Question | Format | Draft reference | Evidence / score |
|---|---|---|---|---|
| fireplace_exists / Object Existence | Is there a fireplace in the room with blue patterned upper walls and wooden lower panels? | yes/no | yes | V1/V2 show carved mantel/fireplace; `Y(yes)` |
| fireplace_location / Object Location | Which room contains the fireplace beneath the exposed wooden ceiling? | Letter | B | A) Red-and-gold corridor; B) Blue-patterned room with wooden lower walls; C) White corridor; D) Pale-blue decorative room. V1/V2; `E(B)` |
| radiator_below_window / Spatial Relation | Are there radiators beneath the windows in the blue-patterned room? | yes/no | yes | V3 shows dark radiators below windows; `Y(yes)` |
| blue_room_windows / Object Counting | How many windows are on the long exterior wall of the blue-patterned room? | Count | 3, visual candidate | V3 shows three distinct window openings in one wall; validate wall identity and full extent; `C(3)` |
| hallway_extinguisher / Object Existence | Is there a fire extinguisher in the white corridor? | yes/no | yes | V4 shows red extinguisher against corridor wall near doorway; confirm object identity in close view; `Y(yes)` |
| mirrored_doors / Object Existence | Are there mirrored door panels at an entrance to a white-paneled room? | yes/no | yes, visual candidate | V5 shows reflective multi-panel door surfaces; verify reflections versus transparent glass; `Y(yes)` |
| arched_passage / Object Existence | Is there an arched passage in the pale-blue decorative room? | yes/no | yes | V6 shows broad curved arch between spaces; `Y(yes)` |
| open_white_door / Object State | Is the white door leading from the red-and-gold corridor into a white room open or closed? | Letter | A, visual candidate | A) Open; B) Closed. V7 shows a panel swung away from the opening; verify which leaf belongs to the doorway; `E(A)` |

## Other potential questions

### 9. [Visual Attribute] What color are the wall panels in the corridor with the gilded vaulted ceiling?

- **Options:** A) Green; B) White; C) Red; D) Blue.
- **Answer:** one letter; candidate **C**, `E(C)`.
- **Evidence:** V7/V8 show red panels framed by gold-colored ornamentation.
- **Why separate:** new visual-attribute template; usable after human confirmation, no exact colorimetry implied.

### 10. [Visual Attribute] What is the dominant pattern of the wood floor in the blue-patterned room?

- **Options:** A) Checkerboard; B) Herringbone; C) Plain parallel strips; D) Hexagons.
- **Answer:** one letter; candidate **B**, `E(B)`.
- **Evidence:** V1/V3 show repeated angled parquet strips.
- **Validation:** distinguish herringbone from other angled parquet patterns in higher-resolution view before approving.

### 11. [Room Counting] How many enclosed rooms are in the scanned environment?

- **Answer:** count; `C(reference)`.
- **Ground truth:** pending. Neither 283 mesh chunks nor two navigation islands is a room count.
- **Work needed:** segment walls/openings, count rooms once, and define whether connecting corridors count. Many similar white rooms make view-based deduplication difficult.

### 12. [Room Size] What is the approximate area of the blue-patterned room, in square meters?

- **Answer:** number; `N(area,t,b)`.
- **Ground truth:** pending.
- **Work needed:** identify its interior floor polygon in transformed geometry; use architectural area rather than whole-scene navigable area or view-position span.

### 13. [Object Dimensions] What is the approximate height of the carved fireplace mantel in the blue-patterned room, in meters?

- **Answer:** number; `N(height,t,b)`.
- **Ground truth:** pending. The GLB has no labeled fireplace template with trustworthy extents.
- **Work needed:** mark mantel bottom/top in scene geometry and specify whether the tall ornamental overmantel is included.

### 14. [Passage Clearance] What is the clear width of the arched passage in the pale-blue room, in meters?

- **Answer:** number; `N(width,t,b)`.
- **Ground truth:** pending.
- **Work needed:** measure the horizontal clear width at the robot footprint height, not the maximum width higher in the arch or scan-hole extent.

### 15. [Room Connectivity] How many doorways must be crossed to get from the blue-patterned room to the red-and-gold corridor?

- **Answer:** count; `C(reference)`.
- **Ground truth:** pending. Sampled viewpoints demonstrate both spaces but not a verified minimum doorway graph.
- **Work needed:** trace real openings and connected region topology; validate route from fixed spawn.

### 16. [Spatial Ordering] Order the carved fireplace, the mirrored doorway, and the arched passage by shortest walking distance from the entrance to the red-and-gold corridor.

- **Options:** A) Carved fireplace; B) Mirrored doorway; C) Arched passage.
- **Answer:** ranking; `R(reference)`.
- **Ground truth:** pending.
- **Work needed:** uniquely identify the corridor entrance and target approaches, define robot radius, then compute collision-aware paths. Viewpoint distances are not target distances.

## Reproducible evidence views

Authoring only: Habitat-Sim0.3.3, sampling seed6, 400×300 RGB, horizontal FOV90°, camera 1.30 m above each listed support position. Positions are runtime **Habitat Y-up meters**; headings are yaw around Habitat +Y, not ROS yaw. Direct camera placement is not proof of robot navigation between views.

| View | Sample | Position | Yaw | Evidence |
|---|---|---|---:|---|
| V1 | P6 | `(6.77447,0.10685,1.41867)` | 90° | Fireplace and carved surround |
| V2 | P12 | `(8.37287,0.10685,6.02445)` | 90° | Fireplace, blue upper walls, wooden ceiling |
| V3 | P3 | `(4.06517,0.10685,6.13574)` | 270° | Three exterior windows/radiators and parquet floor |
| V4 | P1 | `(-9.90981,0.10685,7.53723)` | 90° | White corridor and extinguisher |
| V5 | P1 | `(-9.90981,0.10685,7.53723)` | 180° | Reflective door panels |
| V6 | P8 | `(5.47118,0.10685,12.17097)` | 90° | Pale-blue room and arched passage |
| V7 | P9 | `(-0.79372,0.10685,5.76673)` | 270° | Red/gold corridor opening and white door |
| V8 | P13 | `(-0.23592,0.10685,0.96317)` | 180° | Red-and-gold corridor lengthwise |

Temporary authoring files: `/tmp/opencode/nbg_views.py`, `nbg_contact0.jpg`, `nbg_contact1.jpg`, and `nbg_p<sample>_<heading-index>.jpg` under `/tmp/opencode/`. Script adapts `/tmp/opencode/hm3d_cf_views.py`. Re-render from the parameters/hashes above if temporary files disappear. Do not give these privileged viewpoints to an evaluated agent by default.

## Scoring and validation

`E`: `exact(expected_letter, answer.strip().upper())`; `Y`: `exact(expected, yes_no(answer))`; `C`: `exact(expected, first_number(answer))`; `N(r,t,b)`: `numeric(r, first_number(answer), tolerance=t, band=b)`; `R`: `rank_order(expected, ranking(answer))`. Exact scores 1/0; numeric full credit ±t with linear partial to zero at ±b; ranking correct pair fraction. Parser failures/invalid permutations zero. Numerical references and bands stay unresolved until measured, not guessed from nominal building style.

Validate visual labels at the intended camera height (current robot default 0.45 m), repeated-room identity, doors versus reflections, stage holes, and connected navigation. Many furnishings are absent from sampled views, but no negative whole-building object question is approved from those views alone. The names “blue-patterned room” and “pale-blue decorative room” are author-defined visual identifiers, not dataset semantic labels.
